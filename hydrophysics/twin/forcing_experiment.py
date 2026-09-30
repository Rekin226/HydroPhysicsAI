"""Frozen-parameter forcing and lagged-state-update attribution experiment.

Observed forcing makes a retrospective diagnostic hindcast, not an ex-ante forecast.
Every updated trajectory is scored before ingesting that month's heads. No parameters,
datum offsets, or meter-selection thresholds are fitted to the evaluation interval.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .release import atomic_json, sha256

ARMS = ("climatology", "observed_pumping", "observed_weather", "observed_both",
        "climatology_updated", "observed_both_updated")


def rain_daily(frame: pd.DataFrame) -> pd.Series:
    """CWA midnight 24-hour accumulations, assigned to the preceding Taiwan day.

    The 24-hour field must not be summed at 10-minute frequency. Missing midnight
    records may use a complete day of 144 ten-minute increments. The CWA dry spell
    codes -998/-98 mean zero only for the short accumulation field used here.
    """
    dates = pd.to_datetime(frame.datetime, utc=True).dt.tz_convert("Asia/Taipei")
    d = frame.copy()
    d.index = pd.DatetimeIndex(dates)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    midnight = d[(d.index.hour == 0) & (d.index.minute == 0)]
    daily = pd.to_numeric(midnight.Past24hr, errors="coerce")
    daily = daily.where(daily >= 0)
    daily.index = daily.index.tz_localize(None) - pd.Timedelta(days=1)
    short = pd.to_numeric(d.Past10Min, errors="coerce").replace({-998: 0.0, -98: 0.0})
    short = short.where(short >= 0)
    # Intervals ending at midnight belong to the preceding day.
    short.index = short.index.tz_localize(None) - pd.Timedelta(seconds=1)
    complete = short.resample("D").sum(min_count=144).where(short.resample("D").count() == 144)
    return daily.combine_first(complete).sort_index()


def monthly_weather(root: Path, grid, start: str, end: str) -> tuple[np.ndarray, dict]:
    from .calibrate_flow import _idw_field

    dates = pd.date_range(start, end, freq="MS", inclusive="left")
    rf = pd.read_csv("chou-shui-data/data/rf_stations.csv")
    rain = {}
    sources = {}
    for row in rf.itertuples():
        path = root / "rain" / f"{row.rf_num}.parquet"
        series = rain_daily(pd.read_parquet(path))
        monthly = series.resample("MS").mean()
        counts = series.resample("MS").count()
        rain[row.rf_id] = monthly.where(counts >= monthly.index.days_in_month * 0.8)
        sources[f"rain_{row.rf_id}"] = sha256(path)
    rain = pd.DataFrame(rain)
    gw = pd.read_csv("chou-shui-data/data/gw_stations.csv")
    et = {}
    for row in gw.itertuples():
        path = root / "et" / f"{row.st_id}.json"
        data = json.loads(path.read_text())["daily"]
        series = pd.Series(data["et0_fao_evapotranspiration"],
                           index=pd.to_datetime(data["time"]), dtype=float)
        monthly = series.resample("MS").mean()
        et[row.st_id] = monthly.where(series.resample("MS").count() >= monthly.index.days_in_month * .8)
        sources[f"et_{row.st_id}"] = sha256(path)
    et = pd.DataFrame(et)
    r, e = rain.reindex(dates), et.reindex(dates)
    if (r.notna().sum(axis=1) < 10).any() or (e.notna().sum(axis=1) < 30).any():
        raise ValueError("Insufficient weather stations in an evaluation month")
    rxy = rf.set_index("rf_id").loc[r.columns, ["TM_X97", "TM_Y97"]].to_numpy(float)
    exy = gw.set_index("st_id").loc[e.columns, ["TM_X97", "TM_Y97"]].to_numpy(float)
    rain_field = np.stack([_idw_field(grid, rxy, row) for row in r.to_numpy()], axis=-1)
    et_field = np.stack([_idw_field(grid, exy, row) for row in e.to_numpy()], axis=-1)
    field = np.maximum((rain_field - et_field) / 1000, 0)
    if not np.isfinite(field).all():
        raise ValueError("Non-finite observed weather field")
    old = pd.read_csv("chou-shui-data/data/rf_timeseries.csv", index_col=0, parse_dates=True)
    old = old.resample("MS").mean().reindex(rain.index)
    overlap = rain.index.year == 2022
    diff = rain.loc[overlap] - old.loc[overlap, rain.columns]
    report = {"sources": sources, "minimum_rain_gauges_per_month": int(r.notna().sum(axis=1).min()),
              "minimum_et_locations_per_month": int(e.notna().sum(axis=1).min()),
              "rain_2022_overlap_monthly_rate_mae_mm_day": float(np.nanmean(np.abs(diff))),
              "rain_2022_overlap_mean_new_mm_day": float(np.nanmean(rain.loc[overlap])),
              "rain_2022_overlap_mean_old_mm_day": float(np.nanmean(old.loc[overlap, rain.columns])),
              "rain_mean_mm_day": rain_field.mean(axis=0).tolist(),
              "et_mean_mm_day": et_field.mean(axis=0).tolist(),
              "et_source": "Open-Meteo historical reanalysis, GMT daily ET0, same locations as calibration",
              "rain_source": "CWA original 26 gauges; Taiwan calendar-day accumulations"}
    return field, report


def align_weather(reference, historical, future, historical_dates, future_dates,
                  fit_end="2022-01-01"):
    """Match source climatologies on historical weather only, reserving 2022 for QA."""
    fit = historical_dates < pd.Timestamp(fit_end)
    ratios = []
    for month in range(1, 13):
        selected = fit & (historical_dates.month == month)
        if not selected.any():
            raise ValueError("Missing a calendar month in weather source alignment")
        target = reference[:, selected].mean(axis=1)
        source = historical[:, selected].mean(axis=1)
        if ((source <= 1e-12) & (target > 1e-12)).any():
            raise ValueError("Cannot align nonzero recharge to a zero source climatology")
        ratios.append(np.divide(target, source, out=np.zeros_like(target), where=source > 1e-12))
    ratios = np.stack(ratios, axis=-1)
    return future * ratios[:, future_dates.month - 1], ratios


def _reanalysis_field(root: Path, grid, dates):
    from .calibrate_flow import _idw_field

    gw = pd.read_csv("chou-shui-data/data/gw_stations.csv")
    rain, et, sources = {}, {}, {}
    for row in gw.itertuples():
        path = root / f"{row.st_id}.json"
        data = json.loads(path.read_text())["daily"]
        for values, key in ((rain, "precipitation_sum"), (et, "et0_fao_evapotranspiration")):
            daily = pd.Series(data[key], index=pd.to_datetime(data["time"]), dtype=float)
            monthly = daily.resample("MS").mean()
            values[row.st_id] = monthly.where(
                daily.resample("MS").count() >= monthly.index.days_in_month * .8)
        sources[str(path)] = sha256(path)
    rain, et = pd.DataFrame(rain).reindex(dates), pd.DataFrame(et).reindex(dates)
    if (rain.notna().sum(axis=1) < 30).any() or (et.notna().sum(axis=1) < 30).any():
        raise ValueError("Insufficient reanalysis coverage")
    xy = gw[["TM_X97", "TM_Y97"]].to_numpy(float)
    r = np.stack([_idw_field(grid, xy, row) for row in rain.to_numpy()], axis=-1)
    e = np.stack([_idw_field(grid, xy, row) for row in et.to_numpy()], axis=-1)
    recharge = np.maximum((r - e) / 1000, 0)
    if not np.isfinite(recharge).all():
        raise ValueError("Non-finite reanalysis recharge")
    return recharge, sources, int(rain.notna().sum(axis=1).min())


def reanalysis_weather(root: Path, grid, start: str, end: str) -> tuple[np.ndarray, dict]:
    """Weather-only source alignment fixed before any head evaluation."""
    from .calibrate_flow import DEFAULT_PATHS, _load_recharge_field

    dates = pd.date_range("2022-01-01", end, freq="MS", inclusive="left")
    historical_dates = pd.date_range("2012-01-01", periods=132, freq="MS")
    raw, sources, count = _reanalysis_field(root / "reanalysis", grid, dates)
    historical, hist_sources, _ = _reanalysis_field(
        root / "reanalysis_historical", grid, historical_dates)
    old = _load_recharge_field(
        grid, DEFAULT_PATHS["rf_timeseries"], DEFAULT_PATHS["rf_stations"],
        DEFAULT_PATHS["et_npz"], DEFAULT_PATHS["gw_stations"], "2012-01-01", "2023-01-01")
    old = np.asarray(old)
    recharge, ratios = align_weather(old, historical, raw, historical_dates, dates)
    old = old[:, -12:]
    overlap = recharge[:, :12]
    report = {"sources": {**sources, **hist_sources},
              "source": "Open-Meteo rainfall and ET0 reanalysis, GMT days",
              "source_substitution": True, "bias_adjusted": True,
              "alignment_fit_years": [2012, 2021], "alignment_uses_head_observations": False,
              "alignment_ratio_range": [float(ratios.min()), float(ratios.max())],
              "minimum_rain_locations": count,
              "2022_mean_raw_recharge_mm_day": float(raw[:, :12].mean()*1000),
              "2022_raw_recharge_cell_month_mae_mm_day": float(np.mean(np.abs(raw[:, :12]-old))*1000),
              "2022_recharge_cell_month_mae_mm_day": float(np.mean(np.abs(overlap-old))*1000),
              "2022_mean_recharge_old_mm_day": float(old.mean()*1000),
              "2022_mean_recharge_new_mm_day": float(overlap.mean()*1000),
              "2022_fan_month_correlation": float(np.corrcoef(
                  old.mean(axis=0), overlap.mean(axis=0))[0, 1])}
    return recharge[:, dates >= pd.Timestamp(start)], report


def observed_energy(grid, meta: dict, start: str, end: str) -> tuple[np.ndarray, dict]:
    from .calibrate_flow import DEFAULT_PATHS
    from .pumping import aggregate_pumps, clean_census

    pumps = pd.read_parquet(DEFAULT_PATHS["pump_census"])
    train = pd.read_parquet(DEFAULT_PATHS["pump_kwh"],
                            filters=[("datetime", ">=", pd.Timestamp("2012-01-01")),
                                     ("datetime", "<", pd.Timestamp("2023-01-01"))])
    kept, _, _ = clean_census(pumps, train, cap_duty=float(meta["cap_duty"]),
                              t0="2012-01-01", t1="2023-01-01")
    del train
    future = pd.read_parquet(DEFAULT_PATHS["pump_kwh"],
                             filters=[("datetime", ">=", pd.Timestamp(start)),
                                      ("datetime", "<", pd.Timestamp(end))])
    # Freeze the selected meter population. Re-share its energy, without a new duty filter.
    _, allocated, _ = clean_census(kept, future, cap_duty=None, t0=start, t1=end)
    energy, dates = aggregate_pumps(kept, allocated, grid, start, end)
    coverage = allocated.assign(month=pd.to_datetime(allocated.datetime).dt.to_period("M"))
    counts = coverage.groupby("month").pump.nunique()
    if len(counts) != len(dates) or (counts == 0).any():
        raise ValueError("Missing pumping records for an evaluation month")
    report = {"census_sha256": sha256(DEFAULT_PATHS["pump_census"]),
              "kwh_sha256": sha256(DEFAULT_PATHS["pump_kwh"]),
              "retained_pumps": len(kept), "min_monthly_pumps_with_records": int(counts.min()),
              "max_monthly_pumps_with_records": int(counts.max()),
              "selection_window": ["2012-01-01", "2023-01-01"],
              "cap_duty": meta["cap_duty"], "monthly_GWh": (energy.sum(axis=0) / 1e6).tolist()}
    return energy, report


def advance_before_update(state, steps: int, advance, update):
    """Score/store each prior prediction before updating its state from observations."""
    predictions = []
    for month in range(steps):
        prior = advance(state, month)
        predictions.append(prior.clone())
        state = update(prior, month)
    return torch.stack(predictions, dim=-1)


def paired_metrics(observed: np.ndarray, predictions: dict, minimum_months=12) -> dict:
    valid = np.isfinite(observed)
    for values in predictions.values():
        valid &= np.isfinite(values)
    keep = valid.sum(axis=1) >= minimum_months
    valid[~keep] = False
    if not valid.any():
        raise ValueError("No wells have enough matched evaluation months")
    report = {"stations": int(keep.sum()), "well_months": int(valid.sum()), "metrics": {}}
    for name, values in predictions.items():
        residual = np.where(valid, values - observed, np.nan)
        per_well_mse = np.nansum(residual**2, axis=1)[keep] / valid.sum(axis=1)[keep]
        report["metrics"][name] = {"rmse_m": float(np.sqrt(per_well_mse.mean())),
                                    "bias_m": float(np.nanmean(residual))}
    return report


def run(args) -> dict:
    from .calibrate_flow import set_compile_matvec
    from .drift_diag import fair_baselines
    from .forward import (
        build_model,
        future_forcing,
        load_members,
        member_datum,
        nudge_to_observations,
        rollout,
    )
    from .inputs import input_options, load_twin_inputs
    from .scenario import BASELINE

    protocol = json.loads(args.protocol.read_text())
    start, end = protocol["start"], protocol["end_exclusive"]
    paths = [str(args.theta_dir / name) for name in protocol["parameter_files"]]
    for path in paths:
        if sha256(path) != protocol["parameter_files"][Path(path).name]:
            raise ValueError("Frozen parameters changed after protocol registration")
    if sha256(args.reference) != protocol["reference_forecast_sha256"]:
        raise ValueError("Reference forecast changed after protocol registration")
    members = load_members(paths)
    meta = members[0].meta
    if meta.get("sw_recharge") or meta.get("eta_classes") or meta["meter_filter"] != "dedupe-cap":
        raise ValueError("This experiment requires the frozen datum-gate input construction")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    set_compile_matvec(args.compile_matvec)
    inp = load_twin_inputs(dx=float(meta["dx"]),
                           meter_filter=meta["meter_filter"], cap_duty=meta["cap_duty"],
                           **input_options(meta))
    dates = pd.date_range(start, end, freq="MS", inclusive="left")
    horizon, fit = len(dates), len(inp.dates)
    if dates[0] != inp.dates[-1] + pd.offsets.MonthBegin():
        raise ValueError("Experiment must start immediately after the frozen forecast origin")
    cache = args.out / "forcing.npz"
    if cache.exists() and (args.out / "forcing_audit.json").exists():
        audit = json.loads((args.out / "forcing_audit.json").read_text())
        if audit.get("protocol_sha256") != sha256(args.protocol):
            raise ValueError("Cached forcing predates the current protocol")
        if audit.get("forcing_sha256") != sha256(cache):
            raise ValueError("Cached forcing differs from its audit")
        with np.load(cache) as z:
            if list(z["dates"]) != list(dates.astype(str)):
                raise ValueError("Cached forcing belongs to another interval")
            actual_e, actual_r = z["energy"], z["recharge"]
    else:
        actual_e, pumping = observed_energy(inp.grid, meta, start, end)
        weather_loader = (reanalysis_weather if protocol.get("weather_source") == "openmeteo"
                          else monthly_weather)
        actual_r, weather = weather_loader(args.weather, inp.grid, start, end)
        np.savez_compressed(cache, dates=dates.astype(str).to_numpy(dtype=str),
                            energy=actual_e, recharge=actual_r)
        atomic_json(args.out / "forcing_audit.json", {
            "pumping": pumping, "weather": weather, "protocol_sha256": sha256(args.protocol),
            "forcing_sha256": sha256(cache)})
    cli_e, cli_r, _ = future_forcing(inp, BASELINE, horizon)
    atomic_json(args.out / "forcing_comparison.json", {
        "dates": dates.strftime("%Y-%m").tolist(),
        "actual_pumping_GWh": (actual_e.sum(axis=0) / 1e6).tolist(),
        "climatology_pumping_GWh": (cli_e.sum(dim=0).numpy() / 1e6).tolist(),
        "actual_recharge_mm_day": (actual_r.mean(axis=0) * 1000).tolist(),
        "climatology_recharge_mm_day": (cli_r.mean(dim=0).numpy() * 1000).tolist()})
    observations = pd.read_csv(args.observations, dtype={"station_id": str}, parse_dates=["date"])
    obs = observations.pivot(index="station_id", columns="date", values="head_m").reindex(
        index=inp.sids, columns=dates).to_numpy(float)
    future_input = replace(inp, obs_h=obs, obs_h_filled=obs, dates=dates, ic_month0_filled=False)
    h0 = inp.initial_heads(0).to(device)
    ground = inp.ground_elev.to(device)
    idx, layer = inp.obs_idx, inp.obs_layer
    hist_e, hist_r = inp.E_total[:, 1:].to(device), inp.recharge_field[:, 1:].to(device)
    forcing = {
        "climatology": (cli_e.to(device), cli_r.to(device)),
        "observed_pumping": (torch.tensor(actual_e, device=device), cli_r.to(device)),
        "observed_weather": (cli_e.to(device), torch.tensor(actual_r, device=device)),
        "observed_both": (torch.tensor(actual_e, device=device), torch.tensor(actual_r, device=device)),
    }
    rng = np.random.default_rng(protocol["ic_noise"]["seed"])
    noises = [None, rng.normal(0, protocol["ic_noise"]["sigma_m"], len(inp.sids))]
    sums = {key: np.zeros_like(obs) for key in ARMS}
    hist_sum = np.zeros_like(inp.obs_h)
    completed = 0
    hindcast_audit_path = args.out / "hindcast_audit.json"
    hindcast_audit = (json.loads(hindcast_audit_path.read_text())
                      if hindcast_audit_path.exists() else None)
    if hindcast_audit and hindcast_audit["parameter_hashes"] != protocol["parameter_files"]:
        raise ValueError("Historical replay cache has different parameters")
    for mi, member in enumerate(members):
        stamp = time.perf_counter()
        model, scalars, _ = build_model(inp.grid, member, device)
        if "delay_Sd" in scalars or "aqt_Sa" in scalars:
            raise ValueError("Slow-store state must be explicitly carried by any extended experiment")
        datum = member_datum(member, inp.sids)
        if hindcast_audit:
            path = args.out / f"hindcast_{mi:02d}.npz"
            if sha256(path) != hindcast_audit["files"][path.name]:
                raise ValueError("Historical replay cache changed after its audit")
            with np.load(path) as z:
                hist = torch.tensor(z["heads"], device=device)
            if hist.shape != (h0.shape[0], inp.grid.n_active, fit):
                raise ValueError("Historical replay cache has incompatible dimensions")
        else:
            hist = rollout(model, scalars, h0, hist_e, hist_r, ground, apex_from=h0)
        hist_sum += hist[layer, idx].cpu().numpy()
        for ni, noise in enumerate(noises):
            initial = nudge_to_observations(inp, hist[..., -1], fit - 1, 1.0,
                                             noise=noise, taper_km=5.0, datum=datum)
            outputs = {}
            for arm, (energy, recharge) in forcing.items():
                heads = rollout(model, scalars, initial, energy, recharge, ground,
                                 month0=dates[0].month - 1, apex_from=h0)
                outputs[arm] = heads[layer, idx, 1:].cpu().numpy()
            for arm in ("climatology_updated", "observed_both_updated"):
                energy, recharge = forcing[arm.removesuffix("_updated")]

                def advance(state, t, energy=energy, recharge=recharge,
                            model=model, scalars=scalars):
                    return rollout(model, scalars, state, energy[:, t:t + 1],
                                   recharge[:, t:t + 1], ground,
                                   month0=dates[t].month - 1, apex_from=h0)[..., -1]

                def update(state, t, datum=datum):
                    return nudge_to_observations(future_input, state, t, 1.0,
                                                 taper_km=5.0, datum=datum)

                heads = advance_before_update(initial, horizon, advance, update)
                outputs[arm] = heads[layer, idx].cpu().numpy()
            np.savez_compressed(args.out / f"member_{completed:02d}.npz", **outputs)
            for arm, values in outputs.items():
                sums[arm] += values
            completed += 1
            print(f"Finished {member.label}, initial field {ni}; {completed}/{len(members)*2}", flush=True)
        print(f"Member elapsed {time.perf_counter()-stamp:.1f}s", flush=True)
    hist_mean = hist_sum / len(members)
    offsets = np.nanmean(inp.obs_h - hist_mean, axis=1)
    predicted = {key: values / completed + offsets[:, None] for key, values in sums.items()}
    train = np.concatenate([inp.obs_h, np.full_like(obs, np.nan)], axis=1)
    for key, values in fair_baselines(train, fit, month_offset=0).items():
        predicted[key] = values[:, fit:]
    with np.load(args.reference) as old:
        reference = old["heads_mean"][0, layer, idx, fit:fit + horizon]
        historical_difference = float(np.max(np.abs(
            hist_mean - old["heads_mean"][0, layer, idx, :fit])))
    difference = float(np.max(np.abs(sums["climatology"] / completed - reference)))
    report = paired_metrics(obs, predicted)
    report.update(schema=1, protocol_sha256=sha256(args.protocol),
                  implementation_sha256=sha256(Path(__file__)),
                  observations_sha256=sha256(args.observations),
                  forcing_sha256=sha256(cache), numerical_replication_max_abs_m=difference,
                  historical_replication_max_abs_m=historical_difference,
                  numerical_replication_passed=max(difference, historical_difference)
                      <= protocol["replication_tolerance_m"],
                  kind="retrospective-forcing-attribution", prospective_validation=False,
                  updates_scored_before_assimilation=True, members=completed,
                  period=[start, end], parameter_hashes=protocol["parameter_files"])
    report["weather_source"] = protocol.get("weather_source", "cwa")
    report["limitations"] = protocol["limitations"]
    report["first_month_update_difference_m"] = max(
        float(np.max(np.abs(predicted[key][:, 0] - predicted[key + "_updated"][:, 0])))
        for key in ("climatology", "observed_both"))
    if report["first_month_update_difference_m"] > 1e-8:
        raise ValueError("A state update changed the prediction preceding its first observation")
    rolling = np.concatenate([inp.obs_h[:, -1:], obs[:, :-1]], axis=1)
    report["one_step_comparison"] = paired_metrics(obs, {
        "observed_both_updated": predicted["observed_both_updated"],
        "climatology_updated": predicted["climatology_updated"],
        "persistence_previous_month": rolling,
        "clim": predicted["clim"]})
    np.savez_compressed(args.out / "predictions.npz", **predicted, observed=obs,
                        dates=dates.astype(str).to_numpy(dtype=str), sids=np.array(inp.sids),
                        layers=layer, offsets=offsets)
    report["predictions_sha256"] = sha256(args.out / "predictions.npz")
    atomic_json(args.out / "report.json", report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--theta-dir", type=Path, default=Path("results/twin_runs/stage3_datum_gate"))
    ap.add_argument("--reference", type=Path, default=Path("results/twin_forward/datum_gate.npz"))
    ap.add_argument("--weather", type=Path, default=Path("data_fetch_api/forcing_experiment"))
    ap.add_argument("--observations", type=Path,
                    default=Path("results/twin/new_data_challenge/observations.csv"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compile-matvec", action="store_true")
    args = ap.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    if not result["numerical_replication_passed"]:
        raise SystemExit("Control does not reproduce the frozen forecast; attribution is not accepted")


if __name__ == "__main__":
    main()
