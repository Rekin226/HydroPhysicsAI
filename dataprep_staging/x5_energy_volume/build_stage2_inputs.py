"""Stage-2 rung inputs for the X5 conversion fix (design-time data build; census only).

The twin cleans the census inside ``calibrate_flow`` (``pumping.clean_census``, duty cap 1.0
on 2012-01..2022-12). Scaling kWh before that step would push more meters over the cap and
change the kept set, so the fix is applied AFTER cleaning:

1. run ``pumping.clean_census`` (read-only import) on ``model_census/v1`` exactly as Q0
   does (cap 1.0, t0 2012-01-01, t1 2023-01-01);
2. write the kept pumps with a unique meter key each (``電號1`` = sid), so that
   ``--meter-filter none`` reads them unchanged, and their re-shared kWh with the
   irrigation class multiplied by m(county) and every other class unchanged.

Arms: ``m1`` (identity; the equivalence check against Q0's forcing), ``county`` (primary:
m = a_kept Jan-May 2011-22 / c_twin per county, from the step0_ab report), ``township``
(declared alternative: the county m times a township factor shrunk to the county mean, see
``township_factors``). Outputs go to results/dataprep/x5_energy_volume/stage2_inputs/<arm>/.

Usage: python -I build_stage2_inputs.py m1|county|township
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from x5_energy_volume import x5const as K  # noqa: E402

MC = K.DATAPREP / "pump_electricity" / "model_census" / "v1"
OUT = K.OUT / "stage2_inputs"
STEP0 = K.REPORT_DIR["step0_ab"] / "report.json"
KAPPA = 0.5                 # shrinkage of the township log-factor toward the county mean
CAP = (1 / 3, 3.0)          # bounds on the township factor before renormalising


def county_multipliers() -> dict:
    s = json.loads(STEP0.read_text(encoding="utf-8"))["counties"]
    return {cn: s[cn]["a_m3_per_kwh"]["kept"]["janmay_2011_22"]
            / s[cn]["step0"]["c_twin_m3_per_kwh"] for cn in K.COUNTIES}


LIN_TOWNS = K.DATAPREP / "x3_well_depths" / "evidence" / "v1" / "township_pumping_by_use.csv"


def township_factors() -> dict:
    """{(county, town2): factor}: Lin (2012) Table 2-2 township irrigation volume (X3
    evidence v1, a registered product) / census kept irrigation kWh 2007-08 of the town (mean
    per year), over the county's pooled ratio, log-shrunk by KAPPA and capped; the caller
    renormalises so the county's kWh-weighted mean factor is 1."""
    from x5_energy_volume import census as C
    t = pd.read_csv(LIN_TOWNS)
    t["town2"] = t["town"].astype(str).str[:2]
    rows = []
    for cn in K.COUNTIES:
        p = C.load_pumps(cn)
        E = C.bases(p, C.load_kwh(cn, len(p)))["kept"]
        e = C.per_pump_window(E, 2007, 2008)
        sel = (p["purpose_class"] == "irrigation").to_numpy()
        g = pd.Series(e[sel]).groupby(p.loc[sel, "town"].astype(str).str[:2].to_numpy()).sum()
        rows.append(pd.DataFrame({"county": cn, "town2": g.index, "kwh_0708": g.to_numpy()}))
    k = pd.concat(rows)
    j = t.merge(k, on=["county", "town2"], how="inner")
    j = j[(j["irrigation"] > 0) & (j["kwh_0708"] > 0)]
    out = {}
    for cn, g in j.groupby("county"):
        pooled = g["irrigation"].sum() / g["kwh_0708"].sum()
        f = np.clip(np.exp(KAPPA * np.log(g["irrigation"] / g["kwh_0708"] / pooled)), *CAP)
        out.update({(cn, tw): float(v) for tw, v in zip(g["town2"], f, strict=True)})
    return out


def build(arm: str) -> dict:
    from hydrophysics.twin import pumping
    from hydrophysics.twin.scenario import purpose_class
    pumps = pd.read_parquet(MC / "census.parquet")
    kwh = pd.read_parquet(MC / "kwh.parquet")
    kp, kk, rep = pumping.clean_census(pumps, kwh, cap_duty=1.0, t0="2012-01-01",
                                       t1="2023-01-01")
    kp = kp.copy()
    kp["電號1"] = kp["sid"].astype(str)
    cls = kp["PURPOSE"].map(purpose_class)
    m = pd.Series(1.0, index=kp.index)
    info = {"arm": arm, "n_pumps_kept": int(len(kp)),
            "kept_GWh_2012_2022": float(kk["electricity_kwh"].sum() / 1e6)}
    if arm in ("county", "township"):
        cm = county_multipliers()
        info["county_multiplier"] = cm
        for cn, v in cm.items():
            m[(kp["county"] == cn) & (cls == "irrigation")] = v
    if arm == "township":
        tf = township_factors()
        info["township_factors"] = {f"{c}:{t}": v for (c, t), v in sorted(tf.items())}
        l1 = pd.read_parquet(K.CENSUS_L1, columns=["sid", "uid", "town"])
        tmap = {**dict(zip(l1["sid"].astype(str), l1["town"].astype(str), strict=True)),
                **dict(zip(l1["uid"].astype(str), l1["town"].astype(str), strict=True))}
        town = kp["sid"].astype(str).map(tmap).fillna("").str[:2]
        f = pd.Series([tf.get((c, t), 1.0) for c, t in zip(kp["county"], town, strict=True)],
                      index=kp.index)
        e = kk.groupby("pump")["electricity_kwh"].sum().reindex(kp["sid"]).fillna(0).to_numpy()
        for cn in K.COUNTIES:
            sel = ((kp["county"] == cn) & (cls == "irrigation")).to_numpy()
            w = e[sel]
            f[sel] = f[sel] / (np.sum(f[sel] * w) / max(w.sum(), 1e-9))
        m = m * f.where(cls == "irrigation", 1.0)
        info["township_kappa"] = KAPPA
    mk = kk["pump"].map(pd.Series(m.to_numpy(), index=kp["sid"].astype(str)))
    kk = kk.assign(electricity_kwh=kk["electricity_kwh"] * mk.to_numpy())
    d = OUT / arm
    d.mkdir(parents=True, exist_ok=True)
    kp.to_parquet(d / "census.parquet", index=False)
    kk.to_parquet(d / "kwh.parquet", index=False)
    info["files"] = {f: {"path": rel_to_repo(d / f), "sha256": sha256_file(d / f)}
                     for f in ("census.parquet", "kwh.parquet")}
    (d / "build_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("m1", "county", "township"))
    a = ap.parse_args()
    print(json.dumps(build(a.arm), indent=1, ensure_ascii=False))
