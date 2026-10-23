"""Initial-condition field against the twin's actual construction (WP H2 repair round 3).

The registered ``H2.ic.lowo`` (H2.fill_acceptance.v1) compared the GP field with plain
per-layer IDW. The calibrated stage-3 runs build the initial condition differently (verifier,
2026-10-14): every well's month-0 value of the backfilled series (``no_backfill = False``: a
well that starts later contributes its first observed head) goes into the per-layer IDW, and
the proximal zones are overwritten by a merged-aquifer IDW (``ic_merged_proximal = True``,
zones 205,182,208). This module scores, leave one well out (LOWO), at the held-out well's
model cell (the cell value the twin starts from) and at the well location:

* ``twin``      the twin's construction, computed with the twin's own functions
  (``r3_common.TwinIC``) on the backfilled information set;
* ``twin_obs``  the same with the wells observed in the month only (``--no-backfill``);
* ``gp_obs``    the registered GP field (``ic_field.gp_layered``) on the wells observed in the
  month (the field H2.ic.lowo chose);
* ``gp_bf``     the same GP on the twin's information set (observed wells plus the
  backfilled values of the wells not observed in the month);
* ``idw``       the registered H2.ic.lowo baseline (plain per-layer IDW at the well, observed
  wells only).

Values are the L1 v3 flag-0/1 monthly means of the 158 model wells (C20 mean, else the
partial-month mean: ``s6_ic.jan2012``); the backfill uses the pre-origin record 2012-01..
2022-12 only.

Modes:

* ``jan2012``    post-look (H2.ic.lowo's unit, already scored): LOWO on January 2012. Logs
  a look before computing. Design-time evidence for the choice of the primary candidate of
  the registered confirmatory test.
* ``registered`` the registered test ``H2.ic.twin`` (``H2.repair3.v1``) on January 2013..2022
  (``prereg3.assert_can_score`` first).

Usage: python -I s10_ic_twin.py jan2012|registered
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import ic_field, io_h2  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402

CANDIDATES = ("twin", "twin_obs", "gp_obs", "gp_bf", "idw")


def setup():
    h = io_h2.load("pre", with_daily=False)
    vals = R3.monthly_values(h)
    bf = R3.backfilled(vals)
    m = (h.meta.in_model.to_numpy() & (h.meta.cell.to_numpy() >= 0)
         & np.isfinite(h.meta.dem.to_numpy()) & np.isfinite(h.xy()).all(1))
    idx = np.flatnonzero(m)
    w = io_h2.model_wells().set_index("sid")
    lay0 = w.loc[[h.sids[i] for i in idx], "layer"].to_numpy().astype(int)
    tw = R3.TwinIC()
    xy = h.xy()[idx]
    cells = h.meta.cell.to_numpy()[idx].astype(int)
    sites = R3.site_ids(xy)
    return {"h": h, "idx": idx, "sids": [h.sids[i] for i in idx], "xy": xy, "lay": lay0,
            "cells": cells, "dem": h.meta.dem.to_numpy()[idx], "vals": vals[idx],
            "bf": bf[idx], "tw": tw, "sites": sites, "prox": tw.is_proximal(cells),
            "county": h.meta.county.to_numpy()[idx]}


def lowo_month(S, ym: str) -> pd.DataFrame:
    h, tw = S["h"], S["tw"]
    t = io_h2.month_index(h.dates, ym)
    v_t, bf_t = S["vals"][:, t], S["bf"][:, t]
    xy, lay, cells, dem = S["xy"], S["lay"], S["cells"], S["dem"]
    zone = tw.zone[cells]
    obs = np.isfinite(v_t)
    rows = []
    for i in np.flatnonzero(obs):
        not_i = np.arange(len(v_t)) != i
        tr_o = obs & not_i
        tr_b = np.isfinite(bf_t) & not_i
        c_xy = tw.centroids[cells[i]]
        te = np.vstack([xy[i], c_xy])
        lt, dt = np.array([lay[i]] * 2), np.array([dem[i]] * 2)
        p_go, s_go = ic_field.gp_layered(xy[tr_o], v_t[tr_o], lay[tr_o], dem[tr_o], te, lt, dt)
        if np.array_equal(tr_o, tr_b):
            p_gb, s_gb = p_go, s_go
        else:
            p_gb, s_gb = ic_field.gp_layered(xy[tr_b], bf_t[tr_b], lay[tr_b], dem[tr_b], te,
                                             lt, dt)
        f_b = tw.field(xy[tr_b], bf_t[tr_b], lay[tr_b], cells[tr_b])
        f_o = tw.field(xy[tr_o], v_t[tr_o], lay[tr_o], cells[tr_o])
        pt_b = R3.idw_point_merged(xy[tr_b], bf_t[tr_b], lay[tr_b], xy[[i]], lay[[i]],
                                   zone[tr_b], zone[[i]], S["prox"][[i]])[0]
        pt_o = R3.idw_point_merged(xy[tr_o], v_t[tr_o], lay[tr_o], xy[[i]], lay[[i]],
                                   zone[tr_o], zone[[i]], S["prox"][[i]])[0]
        idw = ic_field.idw_layered(xy[tr_o], v_t[tr_o], lay[tr_o], xy[[i]], lay[[i]])[0]
        rows.append({
            "sid": S["sids"][i], "month": ym, "truth": float(v_t[i]), "layer": int(lay[i]),
            "cell": int(cells[i]), "site": int(S["sites"][i]), "proximal": bool(S["prox"][i]),
            "county": S["county"][i], "n_obs_sources": int(tr_o.sum()),
            "n_bf_sources": int(tr_b.sum()),
            "cell_twin": float(f_b[lay[i], cells[i]]), "cell_twin_obs": float(f_o[lay[i],
                                                                                  cells[i]]),
            "cell_gp_obs": float(p_go[1]), "cell_gp_bf": float(p_gb[1]),
            "sd_cell_gp_obs": float(s_go[1]), "sd_cell_gp_bf": float(s_gb[1]),
            "pt_twin": float(pt_b), "pt_twin_obs": float(pt_o), "pt_gp_obs": float(p_go[0]),
            "pt_gp_bf": float(p_gb[0]), "pt_idw": float(idw),
            "sd_pt_gp_obs": float(s_go[0]), "sd_pt_gp_bf": float(s_gb[0])})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, primary: str | None = None) -> dict:
    out = {"n_cells": int(len(df)), "n_wells": int(df.sid.nunique()),
           "n_sites": int(df.site.nunique()), "months": sorted(df.month.unique().tolist())}
    for lvl in ("cell", "pt"):
        cols = {c: f"{lvl}_{c}" for c in ("twin", "twin_obs", "gp_obs", "gp_bf")}
        if lvl == "pt":
            cols["idw"] = "pt_idw"
        err = {c: (df[col] - df.truth).to_numpy() for c, col in cols.items()}
        res = {"rmse": {c: R3.rmse(e) for c, e in err.items()},
               "rmse_proximal": {c: R3.rmse(e[df.proximal.to_numpy()]) for c, e in err.items()},
               "rmse_non_proximal": {c: R3.rmse(e[~df.proximal.to_numpy()])
                                     for c, e in err.items()},
               "rmse_by_layer": {str(k): {c: R3.rmse(e[(df.layer == k).to_numpy()])
                                          for c, e in err.items()}
                                 for k in sorted(df.layer.unique())},
               "rmse_by_month": {m: {c: R3.rmse(e[(df.month == m).to_numpy()])
                                     for c, e in err.items()}
                                 for m in sorted(df.month.unique())},
               "site_boot_diff_vs_twin_p05_p50_p95": {
                   c: R3.cluster_boot_diff(err[c], err["twin"], df.site) for c in err
                   if c != "twin"},
               "native_cov90_gp": {c: float(np.mean(np.abs(err[c]) <= C.Z90 * df[
                   f"sd_{lvl}_{c}"])) for c in ("gp_obs", "gp_bf")}}
        out[lvl] = res
    if primary is not None:
        e_p = (df[f"cell_{primary}"] - df.truth).to_numpy()
        e_t = (df["cell_twin"] - df.truth).to_numpy()
        r_p, r_t = R3.rmse(e_p), R3.rmse(e_t)
        ci = R3.cluster_boot_diff(e_p, e_t, df.site)
        if r_p < r_t and ci[2] < 0:
            dec = "confirmed"
        elif r_p < r_t:
            dec = "point_only"
        else:
            dec = "not_supported"
        out["decision"] = {"primary": primary, "rmse_primary_cell": r_p, "rmse_twin_cell": r_t,
                           "site_boot_diff_p05_p50_p95": ci, "outcome": dec}
    return out


def record_jan2012_look():
    h = io_h2.load("pre", with_daily=False)
    unit = "H2:ic_lowo_2012-01"
    rec = prereg3.record_look(
        [unit], "heads:2012-01", C.WP, "post_look_repair",
        members={unit: {"description": "leave-one-well-out on the January 2012 L1 heads of "
                        "the model wells (scored under H2.fill_acceptance.v1 as H2.ic.lowo); "
                        "repair round 3 re-scores it against the twin's actual stage-3 "
                        "construction (backfill + merged proximal), post-look",
                        "ids": sorted(s for s, m in zip(h.sids, h.meta.in_model, strict=True)
                                      if m), "window": "2012-01..2012-01"}},
        note="H2 repair round 3 (after the independent verification): post-look LOWO of the "
             "twin's actual initial-condition construction and of the GP field on the twin's "
             "information set, January 2012. Design-time evidence for the choice of the "
             "primary candidate of the confirmatory test H2.ic.twin (registered afterwards "
             "on January 2013..2022).")
    return rec


def main(argv):
    mode = argv[0] if argv else ""
    R3.EV_R3.mkdir(parents=True, exist_ok=True)
    if mode == "jan2012":
        rec = record_jan2012_look()
        S = setup()
        tic = time.time()
        df = lowo_month(S, R3.IC_MONTH0)
        df.to_parquet(R3.EV_R3 / "ic_jan2012_cells.parquet", index=False)
        rep = summarise(df)
        reg = json.loads((C.EVIDENCE_DIR / "registered" / "H2.ic.lowo.json").read_text())
        rep.update({"label": "post-look (H2.ic.lowo's unit, scored 2026-10-13T19:42Z); "
                             "design-time evidence", "look_time": rec["time"],
                    "registered_h2_ic_lowo_rmse": reg["rmse_lowo"],
                    "stage3_recipe": {k: json.loads(R3.STAGE3_SUMMARY.read_text())["cfg"][
                        "recipe"].get(k) for k in ("no_backfill", "ic_merged_proximal",
                                                   "ic_layered_proximal", "zone_boundaries")},
                    "seconds": time.time() - tic})
        R3.jdump(rep, R3.EV_R3 / "ic_jan2012_twin.json")
        print(json.dumps({k: rep[k] for k in ("n_cells", "n_sites")}),
              json.dumps(rep["cell"]["rmse"]), json.dumps(rep["pt"]["rmse"]))
        print("boot cell", json.dumps(rep["cell"]["site_boot_diff_vs_twin_p05_p50_p95"]))
    elif mode == "registered":
        ok = prereg3.assert_can_score(R3.PREREG_ID_R3, C.WP, test_id=R3.TEST_IC)
        doc = ok["prereg"]
        primary = doc["ic_primary_candidate"]
        S = setup()
        tic = time.time()
        df = pd.concat([lowo_month(S, m) for m in R3.IC_MONTHS], ignore_index=True)
        R3.REG_R3.mkdir(parents=True, exist_ok=True)
        df.to_parquet(R3.REG_R3 / f"{R3.TEST_IC}_cells.parquet", index=False)
        rep = summarise(df, primary=primary)
        rep.update({"test_id": R3.TEST_IC, "prereg": R3.PREREG_ID_R3,
                    "assert_can_score": {"sha256": ok["sha256"],
                                         "inputs_verified": ok["inputs_verified"],
                                         "post_registration_looks":
                                             ok["post_registration_looks"],
                                         "unverifiable_prior_looks":
                                             len(ok["unverifiable_prior_looks"])},
                    "pass": rep["decision"]["outcome"] == "confirmed",
                    "note": "decision test: 'pass' means the primary candidate is confirmed "
                            "against the twin's construction (pooled cell-level LOWO RMSE "
                            "lower and the 90 % site-bootstrap interval of the difference "
                            "below 0)", "seconds": time.time() - tic})
        R3.jdump(rep, R3.REG_R3 / f"{R3.TEST_IC}.json")
        print(json.dumps(rep["decision"]), json.dumps(rep["cell"]["rmse"]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
