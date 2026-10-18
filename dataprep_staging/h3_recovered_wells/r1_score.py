"""Repair round 1: registered scoring of ``H3.step_screen`` (``H3.repair.v1``).

Targets: the 15 wells admitted in ``heads/recovered_wells/v1`` and 09030212 (reported only;
its admission does not depend on the screen under ``H3.repair.v1``). For each target the
split is its longest internal C20 gap (09030212: the switch to once-daily readings, post
months = its screened once-daily months of ``H3.snapshot_months_2019_2022.csv``). The null of
each distinct split is the screen applied to every model well (``r1_stepscreen``).

Writes ``H3_evidence/repair1/registered/H3.step_screen.json``, ``..._wells.csv`` and
``..._null.parquet``.

Usage: python -I r1_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import prereg3  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402
from h3_recovered_wells import r1_stepscreen as S  # noqa: E402

PREREG = "H3.repair.v1"
TEST = "H3.step_screen"
RH = "09030212"
OUT = C.EVIDENCE / "repair1" / "registered"


def targets() -> list[str]:
    W = pd.read_csv(C.REPO / "results/dataprep/heads/recovered_wells/v1/recovered_wells.csv",
                    dtype={"sid": str})
    return sorted(W[W.admitted.astype(bool)].sid) + [RH]


def main():
    acs = prereg3.assert_can_score(PREREG, C.WP, test_id=TEST)
    Y = S.c20_matrix()
    meta = inventory.load_meta().set_index("sid")
    model = sorted(inventory.model_wells())
    snap = pd.read_csv(C.EVIDENCE / "registered" / "H3.snapshot_months_2019_2022.csv",
                       dtype={"month": str})
    rows, nulls, cache = [], [], {}
    for sid in targets():
        y = Y[sid].copy() if sid in Y.columns else pd.Series(np.nan, index=S.DATES)
        obs = y.notna().to_numpy()
        if sid == RH:
            g0 = int(np.where(S.DATES == pd.Timestamp("2019-01-01"))[0][0])
            gap = (g0, g0 - 1)
            post_vals = pd.Series(snap.value.to_numpy(float),
                                  index=pd.to_datetime(snap.month + "-01"))
            y = y.copy()
            y.loc[post_vals.index] = post_vals.to_numpy()
            split_kind = "switch to once-daily readings (2019-01); post = screened once-daily " \
                         "months"
        else:
            gap = S.longest_gap(obs)
            split_kind = "longest internal C20 gap"
        pre, post = S.split_months(y.dropna().index, gap)
        r = S.screen_one(Y, y, pre, post, model, meta, sid)
        r.update(split=split_kind, gap_first=S.DATES[gap[0]].strftime("%Y-%m"),
                 gap_last=(S.DATES[gap[1]].strftime("%Y-%m") if gap[1] >= gap[0] else None),
                 gap_months=int(gap[1] - gap[0] + 1))
        if gap not in cache:
            nd = S.null_for_split(Y, gap, model, meta)
            nd["gap_first"], nd["gap_last_idx"] = S.DATES[gap[0]].strftime("%Y-%m"), gap[1]
            cache[gap] = nd
            nulls.append(nd)
        nd = cache[gap]
        nz = nd.loc[nd.testable.astype(bool), "z"].to_numpy(float)
        r["n_null"] = int(len(nz))
        r["null_z_q95"] = float(np.quantile(nz, 0.95)) if len(nz) else None
        r["null_share_z_ge_2"] = float(np.mean(nz >= 2.0)) if len(nz) else None
        if r["testable"] and len(nz) >= S.MIN_NULL:
            r["p"] = S.p_value(r["z"], nz)
            r["flagged"] = bool(r["p"] <= S.ALPHA_LEVEL)
            r["verdict"] = "flagged" if r["flagged"] else "not flagged"
        else:
            r["p"], r["flagged"] = None, None
            r["verdict"] = "untestable" + ("" if r["testable"] else f" ({r.get('reason')})")
        r["role"] = "reported only (H3.repair.v1 rule 3)" if sid == RH else "admission screen"
        rows.append(r)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    N = pd.concat(nulls, ignore_index=True)
    for c in ("neighbours", "neighbour_r", "neighbour_km"):
        N[c] = N[c].map(json.dumps)
    N.to_parquet(OUT / f"{TEST}_null.parquet", index=False)
    Rc = R.copy()
    for c in ("neighbours", "neighbour_r", "neighbour_km"):
        Rc[c] = Rc[c].map(json.dumps)
    Rc.to_csv(OUT / f"{TEST}_wells.csv", index=False)
    adm = R[R.sid != RH]
    res = {
        "test_id": TEST, "prereg": PREREG, "prereg_sha256": acs["sha256"],
        "assert_can_score": {"inputs_verified": acs["inputs_verified"],
                             "post_registration_looks": acs["post_registration_looks"],
                             "unverifiable_prior_looks": len(acs["unverifiable_prior_looks"])},
        "rule": {"alpha": S.ALPHA_LEVEL, "min_null": S.MIN_NULL, "radius_m": S.RADIUS_M,
                 "n_neighbours": S.N_NEIGHBOURS, "min_r": S.MIN_R,
                 "neighbour_pool": "model wells minus the target's site"},
        "admitted_v1": {"n": int(len(adm)), "flagged": sorted(adm[adm.flagged == True].sid),  # noqa: E712
                        "not_flagged": sorted(adm[adm.flagged == False].sid),  # noqa: E712
                        "untestable": sorted(adm[adm.flagged.isna()].sid)},
        "rh_09030212": {k: R[R.sid == RH].iloc[0].get(k) for k in
                        ("verdict", "z", "p", "bias", "rmse_loyo", "n_fit", "n_eval",
                         "neighbours")},
        "per_well": json.loads(R.to_json(orient="records")),
        "null_by_split": {f"{k[0]}:{k[1]}": {
            "gap_first": S.DATES[k[0]].strftime("%Y-%m"),
            "n_testable": int(v.testable.sum()), "n_model_wells": int(len(v)),
            "z_q50": float(v.loc[v.testable.astype(bool), "z"].median()),
            "z_q95": float(v.loc[v.testable.astype(bool), "z"].quantile(0.95)),
            "share_z_ge_2": float((v.loc[v.testable.astype(bool), "z"] >= 2).mean()),
            "bias_mean": float(v.loc[v.testable.astype(bool), "bias"].mean())}
            for k, v in cache.items()},
    }
    (OUT / f"{TEST}.json").write_text(json.dumps(res, indent=1, default=str))
    show = ["sid", "gap_first", "gap_months", "n_fit", "n_eval", "bias", "rmse_loyo", "z", "p",
            "n_null", "verdict"]
    print(R[show].to_string(index=False))
    print(json.dumps(res["admitted_v1"]), json.dumps(res["null_by_split"], indent=0))


if __name__ == "__main__":
    main()
