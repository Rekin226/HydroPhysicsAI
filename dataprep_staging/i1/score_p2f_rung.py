"""The plan's rung-R4 test (WP I1 table: "P2f rerun: model layer-1/2 ratio on local-only slopes
inside the observed 90 % interval") for any forward run, with P2f's own code unchanged.

It imports ``p2_census_repair/p2def/f1_layer_response.py`` and reruns its model side only, for
the declared primary setting (recorded energy arm, 3 km, layer 1 against layer 2): the same
wells, local and regional energy, rain lags, high-pass and OLS, the same site ratios and the
same site bootstrap (seed 2026). The observed side is not recomputed: its interval is read from
``results/dataprep/layer_response/v1/layer_response_localonly.csv`` (P2def.f.layer_response.v1).

    python dataprep_staging/i1/score_p2f_rung.py <forward.npz> [<forward.npz> ...]

Prints per forward run and county the model's median site ratio and its 90 % interval next to
the observed one, and whether the model's median lies inside the observed interval (the R4
test reads Yunlin; Changhua's observed L1 response is not locally identified). Design-time
evidence: every well's 2012-2022 heads were read by earlier WPs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P2DEF = REPO / "dataprep_staging/p2_census_repair/p2def"
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(P2DEF))

import f1_layer_response as F  # noqa: E402

OBS_CSV = REPO / "results/dataprep/layer_response/v1/layer_response_localonly.csv"


def model_ratios(paths: list[str]) -> dict:
    meta = F.station_meta()
    w, _ = F.wells(meta)
    wxy = w[["x_twd97", "y_twd97"]].to_numpy()
    rain = F.rain_for(w)
    w["cell"] = F.cell_of(w["x_twd97"].to_numpy(), w["y_twd97"].to_numpy())
    models = {Path(p).stem: F.model_heads(p) for p in paths}
    px, py, pk, pc, E = F.pump_energy("recorded")
    tot_county = {k: E[(pk == k) & (pc >= 0)].sum(0) for k in F.D.COUNTIES}
    far = E[np.isfinite(px) & np.isfinite(py)].sum(0)[None, :] - np.asarray(
        F.weights(px, py, wxy, F.FAR_KM) @ E)
    loc = np.asarray(F.weights(px, py, wxy, F.PRIMARY_R) @ E)
    rows = []
    for i, wr in enumerate(w.itertuples()):
        kk = "yunlin" if wr.county == F.D.COUNTY_ZH["yunlin"] else "changhua"
        if wr.cell < 0:
            continue
        rec = {"site": wr.site_id, "layer": int(wr.layer), "county": kk}
        for name, Hm in models.items():
            fm = F.fit(Hm[int(wr.layer) - 1, int(wr.cell)], loc[i], rain[i],
                       [tot_county[kk], far[i]])
            if fm:
                rec[f"b_{name}"] = fm[0]
        rows.append(rec)
    df = pd.DataFrame(rows)
    obs = pd.read_csv(OBS_CSV)
    obs = obs[(obs["version"] == "recorded") & (obs["r_km"] == F.PRIMARY_R)
              & (obs["pair"] == "L1/L2") & (obs["heads"] == "obs")].set_index("scope")
    out = {}
    for name in models:
        out[name] = {}
        for scope in ("yunlin", "changhua"):
            st = F.site_ratio_table(df[df["county"] == scope], f"b_{name}", 1)
            med, lo, hi = F.boot_median(st["ratio"].to_numpy(), np.random.default_rng(F.SEED))
            o = obs.loc[scope]
            out[name][scope] = {"model_median_ratio": med, "model_ci90": [lo, hi],
                                "n_sites": int(len(st)),
                                "observed_median_ratio": float(o["median_ratio"]),
                                "observed_ci90": [float(o["ci90_lo"]), float(o["ci90_hi"])],
                                "inside_observed_ci90": bool(
                                    o["ci90_lo"] <= med <= o["ci90_hi"])}
    return out


if __name__ == "__main__":
    print(json.dumps(model_ratios(sys.argv[1:]), indent=1))
