"""Build the WP H2 fill tables (pre-origin and full variants) from L1 v3 with the frozen design.

For every well with L1 data and every month of the variant's window:

* observed C20 months (L1 flag 0/1) are copied (flag 0/1, sd 0, ``sigma_obs_m`` = U2's
  sigma_obs(k) of the month's clean days);
* bridgeable non-C20 months get the daily bridge (flag 2; sd of the bridged-share class);
* an L1 partial month (flag 0/1, 3 <= k < 20 clean days) that the bridge cannot complete
  stays an observation (its L1 flag, method ``observed_partial``, sd 0, sigma_obs(k) from U2's
  rho(k)); it is not an input of the router (the tested configuration);
* every other missing month inside the tested envelope gets the method of its gap class
  (router: neighbour ridge + softImpute, flag 3; GP in months with no observation anywhere,
  flag 4; a design fall-back keeps its own flag: backfill / clim+AR(1) 4, IDW / kriging 3),
  with sd = max(class factor x native sd, 0.05 m); fills after the well's last observation
  (trailing edge, inside the tested 48 months) are flag 6 ("extrapolated beyond the record
  end", plan 2.4 and WP H3), with the method recorded;
* everything else is flag 8 (NaN).

``u2_weight`` = ``sigma_obs^2 / (sigma_obs^2 + sd^2)`` for flags 0-2 (U2; a bridged month is a
completed month, sigma_obs 0.10 m), NaN otherwise. The pre-origin variant also gets the
January 2012 initial-condition fields (``ic_field_2012-01.npz``).

Usage: python -I s5_build.py pre|full OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import scorers  # noqa: E402

from h2_heads_fill import (  # noqa: E402
    bridge,
    ic_field,
    io_h2,
    kriging,
    methods,
    runner,
    scoring,  # noqa: E402
)
from h2_heads_fill import config as C  # noqa: E402

FLAG_OF_METHOD = {"ens_neigh_softimpute": 3, "softimpute_tb": 3, "gp": 4, "backfill": 4,
                  "clim_ar": 4, "clim": 4, "idw": 3, "stk": 3}
PARAMS = C.EVIDENCE_DIR / "design" / "design_params.json"


def build(variant: str, outdir: Path, params_path: Path = PARAMS, mode: str = "fill",
          with_ic: bool = True) -> dict:
    """Write ``outdir/fill.parquet`` (and, pre-origin with ``with_ic``, the January 2012
    initial-condition fields). ``mode`` 'bridge' keeps only the flag 0-2 rows (observations
    and daily-bridged months: the daily-bridge component release)."""
    if mode not in ("fill", "bridge"):
        raise ValueError(mode)
    params = json.loads(Path(params_path).read_text())
    h, Yb, br = runner.load_base(variant)
    W, T = h.Y.shape
    xy, lay = h.xy(), h.meta.layer.to_numpy()
    obs_c20 = np.isfinite(h.Y)
    obs_b = np.isfinite(Yb)
    # bridged months
    bridged = obs_b & ~obs_c20
    bsd = np.full((W, T), np.nan)
    bmeth = np.full((W, T), "", dtype=object)
    bfrac = np.full((W, T), np.nan)
    si = {s: i for i, s in enumerate(h.sids)}
    ti = {d: j for j, d in enumerate(h.dates)}
    if len(br):
        for _, r in br[br.ok].iterrows():
            i, j = si[r.sid], ti[pd.Timestamp(r.month)]
            k = bridge.frac_class(r.frac)
            if k in params.get("bridge_disabled", []):
                raise SystemExit("bridge class disabled by the design: not implemented")
            bsd[i, j] = max(params["bridge_sd_by_class"][str(k)], C.SD_FLOOR)
            bmeth[i, j] = r.method
            bfrac[i, j] = r.frac
    # monthly fills
    info = {}
    mean_r, sd_r, meth_r = methods.router_native(Yb, xy, run_gp=True, info=info)
    cls, runlen, edge = methods.classify_cells(obs_b)
    keys = np.vectorize(lambda c: scoring.class_key(c, params) if c else "")(cls)
    needed = {params["method_by_class"][k] for k in set(keys.ravel()) if k} - {"router"}
    alt = {}
    for m in needed:
        if m == "backfill":
            alt[m] = methods.backfill(Yb)
        elif m == "clim_ar":
            alt[m] = methods.m0_clim_ar(Yb)
        elif m == "idw":
            alt[m] = methods.m6_idw_anom(Yb, xy=xy, layers=lay)
        elif m == "stk":
            alt[m] = kriging.stk_fill(Yb, xy=xy)
    env = np.stack([methods.tested_envelope(obs_b[i]) & (obs_c20[i].sum() >= C.MIN_OBS_MONTHS)
                    for i in range(W)])
    value = np.full((W, T), np.nan)
    sd = np.full((W, T), np.nan)
    flag = np.full((W, T), 8, dtype=np.uint8)
    meth = np.full((W, T), "not_filled_outside_tested_envelope", dtype=object)
    # observed
    value[obs_c20] = h.Y[obs_c20]
    sd[obs_c20] = 0.0
    flag[obs_c20] = h.l1_flag[obs_c20]
    meth[obs_c20] = "observed"
    # bridged
    value[bridged] = Yb[bridged]
    sd[bridged] = bsd[bridged]
    flag[bridged] = 2
    meth[bridged] = bmeth[bridged]
    # monthly fills inside the envelope
    miss = ~obs_b & env
    for i, j in zip(*np.nonzero(miss), strict=True):
        k = keys[i, j]
        m = params["method_by_class"][k]
        ref = params["sd_ref_by_class"][k]
        fac = params["factor_by_class"][k]
        if m == "router":
            v, s0, lab = mean_r[i, j], sd_r[i, j], meth_r[i, j]
        else:
            v, lab = alt[m][0][i, j], m
            s0 = (alt[m][1][i, j] if (ref == m and alt[m][1] is not None) else sd_r[i, j])
        value[i, j] = v
        sd[i, j] = max(fac * s0, C.SD_FLOOR)
        flag[i, j] = 6 if cls[i, j] == "trail" else FLAG_OF_METHOD[lab]
        meth[i, j] = lab
    never = ~obs_b & ~env
    meth[never & (cls == "never")] = "not_filled_no_observation"
    # L1 partial months the bridge could not complete stay observations (U2 sigma_obs(k))
    kc = h.kclean
    opart = h.partial & ~obs_b & (kc >= 3) & np.isfinite(h.partial_mean)
    value[opart] = h.partial_mean[opart]
    sd[opart] = 0.0
    flag[opart] = h.l1_flag[opart]
    meth[opart] = "observed_partial"
    cls = np.where(opart, "", cls)
    runlen = np.where(opart, 0, runlen)
    edge = np.where(opart, 0, edge)
    # sigma_obs and U2 weight
    sig = np.full((W, T), np.nan)
    nd = np.broadcast_to(h.ndays[None, :], (W, T))
    for i, j in zip(*np.nonzero(obs_c20 | opart), strict=True):
        sig[i, j] = scorers.sigma_obs(kc[i, j], nd[i, j])
    sig[bridged] = scorers.SIGMA_OBS_FLOOR
    u2 = np.where(flag <= 2, sig ** 2 / (sig ** 2 + np.nan_to_num(sd) ** 2), np.nan)
    pv = f"{C.PRODUCT_PRE if variant == 'pre' else C.PRODUCT_FULL}/{Path(outdir).name}"
    own = np.isin(meth, ("gp", "backfill", "clim_ar", "clim"))
    src = np.where(np.isin(flag, (0, 1, 2)), "L1v3:own_record",
                   np.where(own, "L1v3:own_record_model",
                            np.where(flag == 8, "", "L1v3:network")))
    rows = pd.DataFrame({
        "sid": np.repeat(np.array(h.sids), T),
        "date": np.tile(np.array([d.strftime("%Y-%m-%d") for d in h.dates]), W),
        "month": np.tile(np.array(h.months), W),
        "value": value.ravel(), "sd": sd.ravel(), "flag": flag.ravel(),
        "method": meth.ravel().astype(str), "gap_class": cls.ravel().astype(str),
        "gap_run_len": runlen.ravel(), "edge_dist": edge.ravel(),
        "bridged_share": bfrac.ravel(), "l1_n_clean_days": kc.ravel(),
        "sigma_obs_m": sig.ravel(), "u2_weight": u2.ravel(),
        "in_model": np.repeat(h.meta.in_model.to_numpy(), T),
        "layer": np.repeat(h.meta.layer.to_numpy(), T),
        "county": np.repeat(h.meta.county.to_numpy(), T),
        "product_version": pv, "source_ids": src.ravel()})
    rows["head_m"] = rows["value"]
    rows["head_sd_m"] = rows["sd"]
    if mode == "bridge":
        rows = rows[rows.flag.isin([0, 1, 2])].reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(outdir / "fill.parquet", index=False)
    stats = summarize(rows, variant, info)
    stats.update({"mode": mode, "params_file": str(params_path)})
    if variant == "pre" and with_ic:
        stats["ic"] = build_ic(h, outdir)
    return stats


def summarize(rows, variant, info):
    gaps = rows[rows.flag != 0]
    gaps = gaps[gaps.flag != 1]
    out = {"variant": variant, "rows": len(rows), "wells": int(rows.sid.nunique()),
           "months": int(rows.month.nunique()), "softimpute": info,
           "by_flag": rows.flag.value_counts().sort_index().to_dict(),
           "by_flag_model_wells": rows[rows.in_model].flag.value_counts().sort_index().to_dict(),
           "by_method": rows.method.value_counts().to_dict(),
           "by_class_model_wells": pd.crosstab(gaps[gaps.in_model].gap_class,
                                               gaps[gaps.in_model].flag).to_dict(),
           "sd_quantiles_by_flag": {int(f): g.sd.quantile([0.1, 0.5, 0.9]).round(3).tolist()
                                    for f, g in rows[rows.flag.between(2, 6)].groupby("flag")}}
    if variant == "full":
        p = rows[rows.month >= "2023-01"]
        out["post_origin_model_wells_by_flag"] = p[p.in_model].flag.value_counts(
        ).sort_index().to_dict()
        fm = rows[rows.month.isin(["2026-02", "2026-03"]) & rows.in_model]
        out["febmar_2026_model_wells_by_flag"] = fm.flag.value_counts().sort_index().to_dict()
    return out


def build_ic(h, outdir):
    """January 2012 initial-condition fields per layer on the model grid (pre-origin)."""
    from h2_heads_fill.s6_ic import jan2012
    g, dem = io_h2.grid_and_dem()
    sids, xy, v, lay, dwell = jan2012(h)
    cx = g.centroids()
    A = len(cx)
    fields = {}
    for name in ("idw", "ked", "gp"):
        val = np.zeros((4, A))
        sdv = np.full((4, A), np.nan)
        for k in range(4):
            lk = np.full(A, k)
            if name == "idw":
                val[k] = ic_field.idw_layered(xy, v, lay, cx, lk)
            elif name == "ked":
                val[k], sdv[k] = ic_field.ked_layered(xy, v, lay, dwell, cx, lk, dem)
            else:
                val[k], sdv[k] = ic_field.gp_layered(xy, v, lay, dwell, cx, lk, dem)
        fields[name] = (val, sdv)
    reg = json.loads((C.EVIDENCE_DIR / "registered" / "H2.ic.lowo.json").read_text())
    np.savez(outdir / "ic_field_2012-01.npz",
             idw=fields["idw"][0], ked=fields["ked"][0], ked_sd=fields["ked"][1],
             gp=fields["gp"][0], gp_sd=fields["gp"][1], decision=np.array(reg["decision"]),
             well_sids=np.array(sids), well_values=v, well_layers=lay,
             cell_centroids=cx, dem=dem)
    return {"decision": reg["decision"], "n_wells": len(sids), "rmse_lowo": reg["rmse_lowo"]}


def main(argv):
    variant, outdir = argv[0], Path(argv[1])
    stats = build(variant, outdir)
    scoring.jdump(stats, outdir.parent / f"build_stats_{variant}.json")
    print(json.dumps({k: v for k, v in stats.items() if k in ("by_flag", "by_flag_model_wells",
                                                              "ic")}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
