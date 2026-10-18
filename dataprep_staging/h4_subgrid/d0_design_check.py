"""WP H4 design-time check, run BEFORE the pre-registration. Reads NO head data.

Plumbing and magnitude checks of the operator: pump counts within 1 and 2 km of each well,
kernel shapes (periodic Theis vs Thiem), the size of the unit-multiplier corrections
(regressors), the kWh -> m3/day conversion against the engine's own abstraction rates, and the
cost of one placebo draw. Writes results/dataprep/reports/H4_evidence/design_time/d0.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

OUT = DATAPREP / "reports/H4_evidence/design_time"


def q(x, p=(0, 10, 25, 50, 75, 90, 100)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {str(k): float(v) for k, v in zip(p, np.percentile(x, p), strict=True)} if x.size else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = L.build_data(L.PRIMARY, with_heads=False)
    t_load = time.perf_counter() - t0
    res = {"config": L.PRIMARY.as_dict(), "reads_head_data": False, "notes": d.notes,
           "n_wells": int(len(d.sids)), "n_pumps_with_xy": int(len(d.pump_xy)),
           "engine_scalars": d.engine_scalars, "t_load_s": round(t_load, 1)}
    from scipy.spatial import cKDTree
    tree = cKDTree(d.pump_xy)
    n1 = np.array([len(tree.query_ball_point(x, 1000.0)) for x in d.xy])
    n2 = np.array([len(tree.query_ball_point(x, 2000.0)) for x in d.xy])
    res["pumps_within_1km"] = {c: q(n1[d.county == c]) for c in ("YL", "CH")}
    res["pumps_within_2km"] = {c: q(n2[d.county == c]) for c in ("YL", "CH")}
    res["wells_by_county_layer_weight"] = {
        f"{c}_L{lay}": {"n": int(((d.county == c) & (d.layer == lay)).sum()),
                        "w": float(d.w[(d.layer == lay)][0]) if (d.layer == lay).any() else None}
        for c in ("YL", "CH") for lay in range(4)}
    res["wells_T_at_bound"] = int(d.t_at_bound.sum())
    res["wells_T_at_bound_with_w"] = int((d.t_at_bound & (d.w > 0)).sum())
    # kernel shapes for the median layer-1 and layer-0 well
    shapes = {}
    for lay in (0, 1):
        i = np.flatnonzero(d.layer == lay)
        i = i[np.argsort(d.T[i])[len(i) // 2]]
        r = np.array([100, 250, 500, 1000, 1500, 1999.0])
        gs, gl = L.kernel_values(r, d.T[i], d.S[i], 1.0, L.PRIMARY)
        shapes[f"layer{lay}"] = {"T": float(d.T[i]), "S": float(d.S[i]),
                                 "kappa_per_km": float(1000 * np.sqrt(L.OMEGA * d.S[i] / d.T[i])),
                                 "r_m": r.tolist(), "g_seasonal": gs.tolist(),
                                 "g_slow": gl.tolist(), "ratio_s_over_l": (gs / gl).tolist()}
    res["kernel_shapes_unit_w"] = shapes
    # unit-multiplier corrections (m): sd over 2012-02..2022-12 per well
    t1 = time.perf_counter()
    Gs, Gl, npump = L.operator_matrices(d)
    Sx, Lx = L.regressors(d, Gs, Gl)
    t_op = time.perf_counter() - t1
    sl = slice(1, None)
    sdS = Sx[:, sl].std(axis=1)
    sdL = Lx[:, sl].std(axis=1)
    sdC = (Sx + Lx)[:, sl].std(axis=1)
    res["support"] = {"wells_with_kernel": int((npump > 0).sum()),
                      "by_county": {c: int(((npump > 0) & (d.county == c)).sum())
                                    for c in ("YL", "CH")}}
    res["unit_correction_sd_m"] = {
        "seasonal": {c: q(sdS[(d.county == c) & (npump > 0)]) for c in ("YL", "CH")},
        "slow": {c: q(sdL[(d.county == c) & (npump > 0)]) for c in ("YL", "CH")},
        "sum": {c: q(sdC[(d.county == c) & (npump > 0)]) for c in ("YL", "CH")}}
    res["unit_correction_mean_m"] = {c: q(Lx[(d.county == c) & (npump > 0), 1:].mean(axis=1))
                                     for c in ("YL", "CH")}
    res["t_operator_s"] = round(t_op, 2)
    # conversion check: per-cell sum of per-pump Q vs the engine's abstraction rate (the engine
    # uses the recorded C0-v1-equivalent census, spread by the calibrated kernel)
    with np.load(L.ENGINE / "engine_arrays.npz", allow_pickle=False) as z:
        Qr = np.array(z["Q_rate"])        # (A, D-1) for months 1..D-1
    Qp = d.qs + d.Qbar                    # (P, D)
    tot_p = Qp[:, 1:].sum(axis=0)
    tot_e = Qr.sum(axis=0)
    res["fan_total_Q_check"] = {"per_pump_sum_over_engine_Q_rate": q(tot_p / tot_e),
                                "note": "per-pump Chow-Lin Yunlin (+ out-of-grid kept pumps) vs "
                                        "the engine's hindcast rates (recorded arm, cached "
                                        "forcing); expected near 1, not equal"}
    # placebo cost
    rng = np.random.default_rng(0)
    t2 = time.perf_counter()
    xy = L.shuffle_xy(d, rng)
    Gs2, Gl2, n2s = L.operator_matrices(d, pump_xy=xy)
    L.regressors(d, Gs2, Gl2)
    res["t_shuffle_draw_s"] = round(time.perf_counter() - t2, 2)
    res["shuffle_support_same"] = bool(np.array_equal(n2s > 0, npump > 0))
    res["shuffle_pumps_within_2km_YL_supported"] = q(n2s[(d.county == "YL") & (npump > 0)])
    t3 = time.perf_counter()
    cen = L.distant_centres(d, rng, npump > 0)
    Gs3, Gl3, n3 = L.operator_matrices(d, centres=cen, support=npump > 0)
    L.regressors(d, Gs3, Gl3)
    res["t_distant_draw_s"] = round(time.perf_counter() - t3, 2)
    res["distant_pumps_within_2km_of_donor"] = q(n3[npump > 0])
    min_d = []
    for i in np.flatnonzero(npump > 0):
        js = Gs3[i].indices
        if js.size:
            min_d.append(np.sqrt(((d.pump_xy[js] - d.xy[i]) ** 2).sum(axis=1)).min())
    res["distant_min_pump_distance_km"] = float(np.min(min_d) / 1000.0)
    (OUT / "d0.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
