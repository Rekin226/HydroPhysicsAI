"""Score frozen model compaction rates at previously unmodeled ring stations.

The protocol and source hashes are checked before loading values. This retrospective
site diagnostic is not a prospective test or evidence of a causal policy response.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..subsidence import _decode_mlcw_name
from .grid import FanGrid
from .release import atomic_json, sha256


def monthly_compaction(wide: pd.DataFrame, pair: tuple[str, str] | None = None) -> pd.Series:
    """Shortening in metres; fix the two physical rings at the first complete survey."""
    if wide.empty:
        raise ValueError("Empty ring series")
    wide = wide.copy()
    wide.index = pd.to_datetime(wide.index, errors="raise")
    if wide.index.has_duplicates:
        raise ValueError("Duplicate ring survey date")
    wide = wide.sort_index()
    complete = wide.dropna(thresh=2)
    if complete.empty:
        raise ValueError("No survey has two ring positions")
    first = complete.iloc[0].dropna()
    if pair is None:
        shallow, deep = first.idxmin(), first.idxmax()
    else:
        shallow, deep = pair
        if shallow not in wide or deep not in wide:
            raise ValueError("Registered reference ring is absent")
        registered = wide[[shallow, deep]].dropna()
        if registered.empty:
            raise ValueError("Registered rings have no paired survey")
        first = registered.iloc[0]
    if shallow == deep or not first[deep] > first[shallow]:
        raise ValueError("Cannot establish shallow and deep reference rings")
    separated = (wide[deep] - wide[shallow]).dropna()
    if separated.empty:
        raise ValueError("Reference rings have no paired surveys")
    # All survey dates are retained until the latest complete survey in each month is
    # chosen; this rule is fixed before scoring and never selects on compaction magnitude.
    separated = separated.groupby(separated.index.to_period("M")).last()
    return (separated.iloc[0] - separated).rename("compaction_m")


def _new_ring(path: Path, pair: tuple[str, str]) -> pd.Series:
    frame = pd.read_parquet(path)
    if set(frame) != {"datetime", "depth", "value"}:
        raise ValueError("Expected dated ring positions with depth and value")
    if frame.duplicated(["datetime", "depth"]).any():
        raise ValueError("Duplicate date/ring measurements")
    values = pd.to_numeric(frame.value, errors="raise")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite ring position")
    wide = frame.assign(value=values).pivot(index="datetime", columns="depth", values="value")
    return monthly_compaction(wide, pair)


def _annual_pairs(series: pd.Series, start: pd.Period | None = None,
                  end: pd.Period | None = None) -> dict[pd.Period, float]:
    out = {}
    for month, value in series.items():
        if (start is not None and month < start) or (end is not None and month > end):
            continue
        earlier = month - 12
        if earlier in series.index and np.isfinite(value) and np.isfinite(series[earlier]):
            out[month] = float((value - series[earlier]) * 100)
    return out


def _historical_baseline(cache_dir: Path, cutoff: pd.Period) -> tuple[float, int, int]:
    site_rates = []
    pairs = 0
    for name in sorted(glob.glob(str(cache_dir / "ls-wra-mlcw-obs__*.parquet"))):
        wide = pd.read_parquet(name)
        wide = wide[pd.to_datetime(wide.index).to_period("M") < cutoff]
        if wide.empty:
            continue
        annual = _annual_pairs(monthly_compaction(wide), end=cutoff - 1)
        if annual:
            site_rates.append(float(np.median(list(annual.values()))))
            pairs += len(annual)
    if not site_rates:
        raise ValueError("No pre-origin ring pairs for the historical baseline")
    return float(np.median(site_rates)), len(site_rates), pairs


def _equal_site_scores(frame: pd.DataFrame, methods: list[str]) -> dict:
    out = {}
    for method in methods:
        error = frame[method] - frame.observed_cm
        site_mse = error.pow(2).groupby(frame.site).mean()
        out[method] = {"rmse_cm_per_year": float(np.sqrt(site_mse.mean())),
                       "bias_cm_per_year": float(error.mean())}
    return out


def evaluate(protocol_path: str | Path) -> dict:
    """Hash-check and score two site-held-out annual ring-rate series."""
    path = Path(protocol_path)
    protocol = json.loads(path.read_text())
    if protocol.get("schema") != 1 or not protocol.get("no_model_refit_or_selection"):
        raise ValueError("Expected a locked, diagnostic-only protocol")
    if protocol.get("measurement_operator") != (
        "fixed registered shallow/deep ring pair; shortening is positive compaction"):
        raise ValueError("Unknown ring measurement operator")
    if protocol.get("monthly_alignment") != (
        "last complete survey per calendar month; model state at same calendar month; "
        "no interpolation"):
        raise ValueError("Unknown monthly alignment")
    files = protocol["files"]
    for key, item in files.items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Source hash changed: {key}")
    cache = Path(protocol["historical_cache_dir"])
    old_files = sorted(cache.glob("ls-wra-mlcw-obs__*.parquet"))
    manifest = protocol["historical_cache_files"]
    if {p.name for p in old_files} != set(manifest):
        raise ValueError("Historical ring cache population changed")
    for old in old_files:
        if sha256(old) != manifest[old.name]:
            raise ValueError("Historical ring cache changed")
    sites = protocol["sites"]
    if len(sites) != len(set(sites)) or not sites:
        raise ValueError("Site list must be nonempty and unique")
    old_sites = {_decode_mlcw_name(str(old)) for old in old_files}
    if set(sites) & old_sites:
        raise ValueError("Evaluation site was already in the old ring cache")
    start = pd.Period(protocol["start_inclusive"], freq="M")
    end = pd.Period(protocol["end_inclusive"], freq="M")
    cutoff = pd.Period(protocol["historical_cutoff_exclusive"], freq="M")
    if not cutoff <= start <= end:
        raise ValueError("Evaluation must start after the historical baseline cutoff")
    historical, training_sites, training_pairs = _historical_baseline(cache, cutoff)
    station_table = pd.read_parquet(files["stations"]["path"]).set_index("sid")
    site_series = {}
    for i, name in enumerate(sites):
        if name not in station_table.index:
            raise ValueError(f"Site absent from station catalog: {name}")
        pair = protocol["ring_pairs"][name]
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("Each site needs two distinct registered rings")
        site_series[name] = _new_ring(Path(files[f"ring_{i}"]["path"]), tuple(pair))
    pairs = []
    for name, series in site_series.items():
        annual = _annual_pairs(series, start=start + 12, end=end)
        if len(annual) < protocol["min_annual_pairs_per_site"]:
            raise ValueError(f"Too few annual pairs at {name}")
        for month, value in annual.items():
            pairs.append({"site": name, "month": month, "year": month.year,
                          "observed_cm": value,
                          "zero_cm": 0.0, "historical_median_cm": historical})
    frame = pd.DataFrame(pairs)
    model_names = [key for key in files if key.endswith("_model")]
    if not model_names:
        raise ValueError("No frozen model provided")
    for model in model_names:
        with np.load(files[model]["path"], allow_pickle=False) as z:
            dates = pd.PeriodIndex(pd.to_datetime(z["dates"]), freq="M")
            if dates.has_duplicates:
                raise ValueError("Duplicate model month")
            if dates[int(z["origin"])] >= start:
                raise ValueError("Model origin must precede the evaluation")
            grid = FanGrid(int(z["nx"]), int(z["ny"]), float(z["dx"]),
                           float(z["x0"]), float(z["y0"]), z["mask"])
            scenarios = [str(x) for x in z["scenario_names"]]
            if "baseline" not in scenarios:
                raise ValueError("Missing unchanged-pumping baseline scenario")
            field = z["subs_mean"][scenarios.index("baseline")]
            lookup = {date: j for j, date in enumerate(dates)}
            values = {}
            for name in sites:
                metadata = station_table.loc[name]
                cell = grid.active_index(float(metadata.X_3826), float(metadata.Y_3826))
                if cell is None:
                    raise ValueError(f"Site outside model grid: {name}")
                values[name] = field[cell]
            predicted = []
            for row in frame.itertuples():
                if row.month not in lookup or row.month - 12 not in lookup:
                    raise ValueError("Model lacks a paired evaluation month")
                m = values[row.site]
                predicted.append(float((m[lookup[row.month]]
                                        - m[lookup[row.month - 12]]) * 100))
            frame[model] = predicted
    methods = ["zero_cm", "historical_median_cm", *model_names]
    site_results = {}
    for name, part in frame.groupby("site"):
        site_results[name] = {"annual_pairs": len(part),
                              "observed_mean_cm_per_year": float(part.observed_cm.mean()),
                              "scores": _equal_site_scores(part, methods)}
    year_results = {str(year): {"annual_pairs": len(part),
                                "sites": int(part.site.nunique()),
                                "scores": _equal_site_scores(part, methods)}
                    for year, part in frame.groupby("year")}
    return {"schema": 1, "task": "retrospective-unmodeled-site-compaction-rate",
            "protocol_sha256": sha256(path), "prospective_validation": False,
            "causal_policy_validation": False,
            "training_ring_sites": training_sites, "training_annual_pairs": training_pairs,
            "historical_median_cm_per_year": historical,
            "evaluation_sites": len(sites), "evaluation_annual_pairs": len(frame),
            "annual_pairs_are_overlapping": True,
            "scores": _equal_site_scores(frame, methods), "by_site": site_results,
            "by_year_exploratory": year_results,
            "limitations": ["Two ring sites absent from the old 14-site clean cache; the regional leveling network was used for column selection",
                            "The data were acquired before this protocol; this is retrospective",
                            "Ring interval compaction is compared with modeled column compaction at one 1-km cell",
                            "Observed intervention effects and predictive coverage are not tested"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.protocol)
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "by_site"}, indent=2))


if __name__ == "__main__":
    main()
