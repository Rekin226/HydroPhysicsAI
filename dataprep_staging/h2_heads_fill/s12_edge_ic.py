"""Initial condition at late-starting wells (WP H2 repair round 3; design-time evidence).

Leave-one-well-out at observed wells (``H2.ic.lowo``, ``s10_ic_twin``) measures the field
between wells. The twin's actual initial condition also covers the cells of wells that start
later: stage 3 puts each such well's first observed head (the backfill) into its IDW, so its
own cell starts near that stale value. This script asks, for January 2012, which value is
closest to the truth at a well whose record is cut to start 66 months later (family E
leading-edge masks, deterministic per well; every H2 realization r = 0..4, 10..19, 30..59,
each well once, first realization in which it appears):

* ``backfill``        the well's first observed head after the masked 66 months (the twin's
  backfill, from the run table);
* ``router``          H2's leading-edge router fill (neighbour ridge + softImpute; the run
  table's native prediction);
* ``twin_own``        the twin's stage-3 field at the well's cell with the well's own backfill
  among the sources (its actual construction for a late-starting well);
* ``twin``            the same field without the well (``s10_ic_twin`` January 2012 LOWO);
* ``gp_obs``          the GP field on the wells observed in January 2012, without the well;
* ``gp_bf_own``       the GP on the twin's information set with the well's own backfill.

Truth: the well's January 2012 C20 value, clean truth only, model wells only. All of these
cells were scored before (E realizations; January 2012 is H2.ic.lowo's unit), so the result
is design-time evidence for I1, never a gate. Logs a look before computing.

Usage: python -I s12_edge_ic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import ic_field, io_h2, scoring  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402
from h2_heads_fill.s10_ic_twin import setup  # noqa: E402

UNITS = ["H2:design_E_r0-4", "H2:fresh_E_r10-19", "H2:fresh2_E_r30-59", "H2:ic_lowo_2012-01"]


def lead_cells():
    rs = list(R3.E_R_DESIGN) + list(R3.E_R_V1) + list(R3.E_R_V2)
    parts = []
    for r in rs:
        d = scoring.load_runs("pre", ["E"], {"E": [r]})
        parts.append(d[(d.stratum == "E_lead66") & (d.t == 0)])
    df = pd.concat(parts, ignore_index=True).sort_values("r")
    return df.drop_duplicates("sid", keep="first").reset_index(drop=True)


def main():
    rec = prereg3.record_look(
        UNITS, "heads:2012-01..2017-06", C.WP, "post_look_repair",
        note="H2 repair round 3: January 2012 leading-edge cells of every H2 family-E "
             "realization (each well once) compared with the twin's backfill, the router "
             "and the IC fields; design-time evidence for I1 (the initial condition at "
             "late-starting wells).")
    S = setup()
    tw = S["tw"]
    t0 = io_h2.month_index(S["h"].dates, R3.IC_MONTH0)
    pos = {s: k for k, s in enumerate(S["sids"])}
    jan = pd.read_parquet(R3.EV_R3 / "ic_jan2012_cells.parquet").set_index("sid")
    E = lead_cells()
    E = E[E.in_model.astype(bool) & E.clean.astype(bool) & E.sid.isin(pos)]
    v_t, bf_t = S["vals"][:, t0], S["bf"][:, t0]
    xy, lay, cells, dem = S["xy"], S["lay"], S["cells"], S["dem"]
    rows = []
    for r in E.itertuples():
        i = pos[r.sid]
        src = np.isfinite(bf_t)
        vals = bf_t.copy()
        vals[i] = r.pred_backfill                    # its own first head after the mask
        f = tw.field(xy[src], vals[src], lay[src], cells[src])
        te = tw.centroids[cells[i]][None, :]
        p, _ = ic_field.gp_layered(xy[src], vals[src], lay[src], dem[src], te, lay[[i]],
                                   dem[[i]])
        j = jan.loc[r.sid] if r.sid in jan.index else None
        rows.append({"sid": r.sid, "r": int(r.r), "truth": float(r.truth),
                     "truth_l1": float(v_t[i]), "layer": int(lay[i]),
                     "proximal": bool(S["prox"][i]), "backfill": float(r.pred_backfill),
                     "router": float(r.pred_router), "twin_own": float(f[lay[i], cells[i]]),
                     "gp_bf_own": float(p[0]),
                     "twin": float(j.cell_twin) if j is not None else np.nan,
                     "gp_obs": float(j.cell_gp_obs) if j is not None else np.nan})
    df = pd.DataFrame(rows)
    df.to_parquet(R3.EV_R3 / "edge_ic_jan2012_cells.parquet", index=False)
    cands = ("backfill", "router", "twin_own", "gp_bf_own", "twin", "gp_obs")
    err = {c: (df[c] - df.truth).to_numpy() for c in cands}
    sites = R3.site_ids(xy[[pos[s] for s in df.sid]])
    out = {"label": "design-time evidence (E cells scored in v1/v2; January 2012 is "
                    "H2.ic.lowo's unit); never a gate", "look_time": rec["time"],
           "n_wells": int(len(df)), "n_sites": int(len(set(sites))),
           "realizations": sorted(df.r.unique().tolist()),
           "truth_matches_l1": bool(np.allclose(df.truth, df.truth_l1)),
           "rmse": {c: R3.rmse(e) for c, e in err.items()},
           "mae": {c: float(np.mean(np.abs(e))) for c, e in err.items()},
           "rmse_proximal": {c: R3.rmse(e[df.proximal.to_numpy()]) for c, e in err.items()},
           "rmse_non_proximal": {c: R3.rmse(e[~df.proximal.to_numpy()])
                                 for c, e in err.items()},
           "n_proximal": int(df.proximal.sum()),
           "site_boot_diff_vs_twin_own_p05_p50_p95": {
               c: R3.cluster_boot_diff(err[c], err["twin_own"], sites) for c in cands
               if c != "twin_own"}}
    R3.jdump(out, R3.EV_R3 / "edge_ic_jan2012.json")
    print(json.dumps({k: out[k] for k in ("n_wells", "n_sites", "rmse", "rmse_non_proximal",
                                          "n_proximal")}, indent=0))
    print(json.dumps(out["site_boot_diff_vs_twin_own_p05_p50_p95"]))


if __name__ == "__main__":
    main()
