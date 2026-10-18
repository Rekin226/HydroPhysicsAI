"""WP P4m item (4): recorded-forcing diagnostic arm 2023-01..2025-06 with Changhua metered
forcing (U4; reported only, never gating), as declared in ``P4m.recorded_arm_2023_25.v1``.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_recorded

The forcing experiment's protocol (``hydrophysics.twin.forcing_experiment``) on the sparse
twin with the cwa theta, one member, noise-free initial field: hindcast 2012-02..2022-12 with
the arm's pre-origin forcing, state nudged to the 2022-12 observations (gain 1, taper 5 km,
the theta's well datum), forecast 2023-01..2025-06 with climatology recharge; prediction =
model + per-well mean(obs - hindcast) over 2012-01..2022-12. Refuses to run unless the P4m
R1 recommendation (pre-origin, U7) has been written and hashed.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime as _dt  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO  # noqa: E402

from p4_changhua import looks_p4m  # noqa: E402
from p4_changhua import p4m_lib as L  # noqa: E402

OUT = L.STAGE / "recorded"
RECOMMENDATION = L.STAGE / "recommendation" / "recommendation_preorigin.json"
FX = REPO / "results" / "twin" / "forcing_experiment"
FX_OBS = REPO / "results" / "twin" / "new_data_challenge" / "observations.csv"
START, END_EXCL = "2023-01-01", "2025-07-01"


def _metrics(obs, preds: dict, sel: np.ndarray, base: str = "clim") -> dict:
    from hydrophysics.twin.forcing_experiment import paired_metrics

    o = obs[sel]
    p = {k: v[sel] for k, v in preds.items()}
    rep = paired_metrics(o, p)
    valid = np.isfinite(o)
    for v in p.values():
        valid &= np.isfinite(v)
    keep = valid.sum(axis=1) >= 12
    valid[~keep] = False
    for k, v in p.items():
        e = np.where(valid, v - o, np.nan)
        mse = np.nansum(e ** 2, axis=1)[keep] / valid.sum(axis=1)[keep]
        rep["metrics"][k]["mean_of_rmse_m"] = float(np.sqrt(mse).mean())
        sd_o = np.array([np.std(o[w, valid[w]]) for w in np.flatnonzero(keep)])
        sd_p = np.array([np.std(v[w, valid[w]]) for w in np.flatnonzero(keep)])
        rep["metrics"][k]["amplitude_ratio_median"] = float(np.median(sd_p / sd_o))
    for k in p:
        rep["metrics"][k]["rmse_ratio_to_clim"] = (rep["metrics"][k]["rmse_m"]
                                                   / rep["metrics"][base]["rmse_m"])
        rep["metrics"][k]["mean_of_rmse_ratio_to_clim"] = (
            rep["metrics"][k]["mean_of_rmse_m"] / rep["metrics"][base]["mean_of_rmse_m"])
    return rep


def main() -> int:
    import torch

    from hydrophysics.twin.drift_diag import fair_baselines
    from hydrophysics.twin.forward import (
        future_forcing,
        member_datum,
        nudge_to_observations,
        rollout,
    )
    from hydrophysics.twin.scenario import BASELINE, climatology

    if not RECOMMENDATION.exists():
        raise SystemExit("write the pre-origin R1 recommendation first (U7)")
    rec_sha = sha256_file(RECOMMENDATION)
    OUT.mkdir(parents=True, exist_ok=True)
    created = _dt.datetime.now(_dt.timezone.utc).isoformat()
    t_all = time.perf_counter()
    S = L.setup()
    inp, tw = S["inp"], S["tw"]
    dates = pd.date_range(START, END_EXCL, freq="MS", inclusive="left")
    horizon = len(dates)
    with np.load(FX / "forcing.npz", allow_pickle=False) as z:
        if [str(d)[:10] for d in z["dates"]] != [str(d.date()) for d in dates]:
            raise SystemExit("cached recorded forcing covers other months")
        e_rec = np.asarray(z["energy"], dtype=np.float64)
    cli_e, cli_r, _ = future_forcing(inp, BASELINE, horizon)
    cli_e, cli_r = cli_e.numpy(), cli_r.numpy()
    Ech, Ech_full, _ = L.changhua_forcing(S)
    Ech_nd, Ech_nd_full, _ = L.changhua_forcing(S, nodomestic=True)
    ch_clim, _ = climatology(Ech_full, inp.dates, horizon)
    ch_rec, mths, info_rec = L.c0_cells("full", "forcing_recorded_arm",
                                        first="2023-01", last="2025-06",
                                        grid_hash_expected=S["grid_hash"])
    ch_rec_nd, _, _ = L.c0_cells("full", "forcing_recorded_arm", nodomestic=True,
                                 first="2023-01", last="2025-06",
                                 grid_hash_expected=S["grid_hash"])
    yl_rec_c0, _, _ = L.c0_cells("full", "forcing_recorded_arm", counties=("yunlin",),
                                 first="2023-01", last="2025-06",
                                 grid_hash_expected=S["grid_hash"])
    obs_df = pd.read_csv(FX_OBS, dtype={"station_id": str}, parse_dates=["date"])
    looks_p4m.record(["p4m.T4.recorded"], ids=sorted(set(obs_df.station_id) & set(S["sids"])),
                     window="2023-01..2025-06",
                     description="2023-01..2025-06 heads of the forcing experiment's "
                                 "observations (registered QC), validator of the "
                                 "recorded-forcing arm only (U4/U7)",
                     purpose="recorded_arm_validator", prereg_id=L.PRE_REC,
                     tag=f"[p4m_recorded:T4:{created}]",
                     note=f"T4 recorded-forcing arm; recommendation sha256 {rec_sha} written "
                          "before this look")
    obs = obs_df.pivot(index="station_id", columns="date", values="head_m").reindex(
        index=list(S["sids"]), columns=dates).to_numpy(float)
    datum = member_datum(S["member"], list(S["sids"]))
    hind = {"A0": L.rollout(S), "A1": L.rollout(S, E_extra=Ech),
            "A2": L.rollout(S, E_extra=Ech_nd)}
    fit = len(inp.dates)
    offs, init = {}, {}
    for k, H in hind.items():
        hist = H[S["ol"], S["oi"], :]                              # (W, 132) 2012-01..
        offs[k] = np.nanmean(inp.obs_h - hist, axis=1)
        init[k] = nudge_to_observations(inp, torch.as_tensor(H[..., -1]), fit - 1, 1.0,
                                        noise=None, taper_km=5.0, datum=datum).numpy()
    arms = {"R_clim_zero": ("A0", cli_e, None), "R_clim_ch": ("A1", cli_e, ch_clim),
            "R_obs_zero": ("A0", e_rec, None), "R_obs_ch": ("A1", e_rec, ch_rec),
            "R_obs_ch_nodomestic": ("A2", e_rec, ch_rec_nd)}
    preds = {}
    for name, (hk, e, ex) in arms.items():
        H = tw.rollout(init[hk], E=e, R=cli_r, E_extra=ex, apex_from=S["h0"])
        preds[name] = H[S["ol"], S["oi"], 1:] + offs[hk][:, None]
    # numerical check of the forecast (R_obs_ch) against the torch rollout
    t0 = time.perf_counter()
    Ht = rollout(S["model"], S["sc"], torch.as_tensor(init["A1"]),
                 torch.as_tensor(e_rec + ch_rec), torch.as_tensor(cli_r),
                 torch.as_tensor(S["ge"]), apex_from=torch.as_tensor(S["h0"])).numpy()
    Hs = tw.rollout(init["A1"], E=e_rec, R=cli_r, E_extra=ch_rec, apex_from=S["h0"])
    num = {"max_abs_dh_m": float(np.abs(Ht - Hs).max()), "t_torch_s":
           round(time.perf_counter() - t0, 1)}
    num["pass"] = bool(num["max_abs_dh_m"] <= 1e-5)
    train = np.concatenate([inp.obs_h, np.full_like(obs, np.nan)], axis=1)
    preds["clim"] = fair_baselines(train, fit, month_offset=0)["clim"][:, fit:]
    G = L.groups(S)
    res = {"all": _metrics(obs, preds, G["all"]), "changhua": _metrics(obs, preds, G["CH"]),
           "yunlin": _metrics(obs, preds, G["YL"])}
    for g in ("changhua", "yunlin", "all"):
        m = res[g]["metrics"]
        res[g]["R_obs_ch_vs_R_obs_zero_rmse_rel_change"] = (m["R_obs_ch"]["rmse_m"]
                                                            / m["R_obs_zero"]["rmse_m"] - 1)
    tot = lambda a: (a.sum(axis=0) / 1e6).tolist()  # noqa: E731
    out = {"wp": "P4m", "prereg": L.PRE_REC, "created": created,
           "recommendation_sha256_before_first_score": rec_sha,
           "protocol": "forcing_experiment on the sparse twin, cwa theta, 1 member, "
                       "noise-free initial field (U4: reported only)",
           "numerical_check_R_obs_ch_vs_torch": num,
           "forcing_GWh_per_month": {"yunlin_recorded_experiment": tot(e_rec),
                                     "yunlin_recorded_c0_full_v3": tot(yl_rec_c0),
                                     "changhua_chiayi_recorded_c0_full_v3": tot(ch_rec),
                                     "changhua_chiayi_climatology": tot(ch_clim),
                                     "yunlin_climatology": tot(cli_e), "months":
                                         [str(d.date())[:7] for d in dates]},
           "c0_full_flags_2023_2025": info_rec["flags"], "guard_notes": info_rec["notes"],
           "scores": res, "t_total_s": round(time.perf_counter() - t_all, 1)}
    L.jdump(out, OUT / "results.json")
    np.savez_compressed(OUT / "predictions.npz", sids=S["sids"], obs=obs,
                        months=np.array([str(d.date())[:7] for d in dates]),
                        **{k: v for k, v in preds.items()})
    m = res["changhua"]["metrics"]
    L.log(f"T4 Changhua rmse: obs_zero {m['R_obs_zero']['rmse_m']:.3f} obs_ch "
          f"{m['R_obs_ch']['rmse_m']:.3f} clim {m['clim']['rmse_m']:.3f}; numerical "
          f"{num['max_abs_dh_m']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
