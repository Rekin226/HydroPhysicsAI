"""Describe saved subsidence scenarios as conditional, paired model experiments.

Annual sidecars support year-end cumulative change and year-on-year compaction rates.
They do not establish causal effects, feasible implementation, or predictive coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .release import atomic_json, sha256


def _summarize(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "max": float(np.max(values)),
            "fraction_positive": float(np.mean(values > 0))}


def evaluate_sidecar(path: str | Path, *, policy_start_year: int = 2026,
                     final_year: int = 2032, late_start_year: int = 2028,
                     forward: str | Path | None = None) -> dict:
    """Pair each policy with its baseline under the same parameter/IC/rheology member."""
    with np.load(path, allow_pickle=False) as z:
        required = {"subs_members_yr", "subs_scale_m", "years", "scenario_names",
                    "member", "ic", "rheology"}
        if required - set(z.files):
            raise ValueError(f"Sidecar keys missing: {sorted(required - set(z.files))}")
        raw = z["subs_members_yr"]
        years = np.asarray(z["years"], dtype=int)
        names = [str(name) for name in z["scenario_names"]]
        members = [str(value) for value in z["member"]]
        ics = np.asarray(z["ic"], dtype=int)
        rheologies = np.asarray(z["rheology"], dtype=str)
        scale_cm = float(z["subs_scale_m"]) * 100
        ye_idx = np.asarray(z["ye_idx"], dtype=int) if "ye_idx" in z else None
    if raw.ndim != 4 or raw.shape[0] != len(names) or raw.shape[1] != len(members):
        raise ValueError("Expected scenario/member/cell/year subsidence array")
    if np.any(np.abs(raw.astype(np.int32)) >= np.iinfo(np.int16).max):
        raise ValueError("Quantized subsidence sidecar may be saturated")
    if raw.shape[3] != len(years) or not np.array_equal(np.diff(years),
                                                        np.ones(len(years) - 1)):
        raise ValueError("Year-end observations must be consecutive")
    if len(ics) != len(members) or len(rheologies) != len(members):
        raise ValueError("Member, initial-condition and rheology labels must align")
    if len(set(zip(members, ics, rheologies, strict=True))) != len(members):
        raise ValueError("Ambiguous paired member labels")
    if len(set(names)) != len(names) or "baseline" not in names:
        raise ValueError("One uniquely named baseline is required")
    if not np.isfinite(scale_cm) or scale_cm <= 0:
        raise ValueError("Invalid subsidence unit scale")
    crosscheck = None
    if forward is not None:
        if ye_idx is None:
            raise ValueError("Annual sidecar needs year-end indices for monthly crosscheck")
        with np.load(forward, allow_pickle=False) as z:
            if names != [str(v) for v in z["scenario_names"]]:
                raise ValueError("Monthly and annual scenarios differ")
            months = np.asarray(z["dates"]).astype("datetime64[M]")
            if not np.array_equal(months[ye_idx].astype("datetime64[Y]").astype(int) + 1970,
                                  years):
                raise ValueError("Annual indices do not match monthly December states")
            if np.any(months[ye_idx].astype(int) % 12 != 11):
                raise ValueError("Annual sidecar index is not December")
            annual_mean = raw.astype(np.float32).mean(axis=1) * (scale_cm / 100)
            monthly_mean = z["subs_mean"][:, :, ye_idx]
            if annual_mean.shape != monthly_mean.shape:
                raise ValueError("Monthly and annual cell arrays differ")
            max_abs = float(np.max(np.abs(annual_mean - monthly_mean)))
            if max_abs > 0.0005:
                raise ValueError("Quantized annual member mean disagrees with monthly field")
        crosscheck = {"monthly_forward_sha256": sha256(forward),
                      "max_annual_mean_difference_m": max_abs,
                      "tolerance_m": 0.0005}
    before, first, last, late_before = (policy_start_year - 1, policy_start_year,
                                        final_year, late_start_year - 1)
    if not (years[0] <= before < first <= late_before < last <= years[-1]):
        raise ValueError("Requested comparison years are outside the annual sidecar")
    ix = {int(year): i for i, year in enumerate(years)}
    # Mean over cells first; the paired member axis remains intact. Values are cm.
    fan = raw.astype(np.float64).mean(axis=2) * scale_cm
    baseline = fan[names.index("baseline")]
    results = {}
    for scenario, name in enumerate(names):
        if name == "baseline":
            continue
        policy = fan[scenario]
        pre_difference = float(np.max(np.abs(
            baseline[:, :ix[before] + 1] - policy[:, :ix[before] + 1])))
        if pre_difference > 1e-8:
            raise ValueError(f"{name} differs from baseline before its declared start")
        avoided = baseline - policy
        year_end = {}
        for year in range(first, last + 1):
            j = ix[year]
            year_end[str(year)] = {
                "avoided_cumulative_cm": _summarize(avoided[:, j]),
                "avoided_annual_rate_cm_per_year": _summarize(avoided[:, j] - avoided[:, j-1]),
            }
        # A late rate effect must be distinguished from a one-time rebound in 2026.
        late_rate = (avoided[:, ix[last]] - avoided[:, ix[late_before]]) / (
            last - late_before)
        late_baseline_rate = (baseline[:, ix[last]] - baseline[:, ix[late_before]]) / (
            last - late_before)
        late_policy_rate = (policy[:, ix[last]] - policy[:, ix[late_before]]) / (
            last - late_before)
        by_rheology = {}
        for label in sorted(set(rheologies)):
            selected = rheologies == label
            by_rheology[label] = {
                f"avoided_{last}_cm": _summarize(avoided[selected, ix[last]]),
                "late_rate_reduction_cm_per_year": _summarize(late_rate[selected]),
            }
        # Cell fractions describe spatial heterogeneity of ensemble-mean model response,
        # not the fraction of people/land protected and not a calibrated probability.
        cell_avoided = (raw[names.index("baseline"), :, :, ix[last]].astype(np.float64)
                        - raw[scenario, :, :, ix[last]].astype(np.float64)).mean(axis=0)
        cell_late_rate = (
            raw[names.index("baseline"), :, :, ix[last]].astype(np.float64)
            - raw[scenario, :, :, ix[last]].astype(np.float64)
            - raw[names.index("baseline"), :, :, ix[late_before]].astype(np.float64)
            + raw[scenario, :, :, ix[late_before]].astype(np.float64)
        ).mean(axis=0) / (last - late_before)
        results[name] = {
            "paired_members": len(members),
            "year_end": year_end,
            "baseline_late_rate_cm_per_year": _summarize(late_baseline_rate),
            "policy_late_rate_cm_per_year": _summarize(late_policy_rate),
            "late_rate_reduction_cm_per_year": _summarize(late_rate),
            "benefit_after_first_year_cm": _summarize(avoided[:, ix[last]]
                                                      - avoided[:, ix[first]]),
            "by_rheology": by_rheology,
            f"fraction_grid_cells_with_positive_{last}_difference": float(
                np.mean(cell_avoided > 0)),
            "fraction_grid_cells_with_positive_late_rate_reduction": float(
                np.mean(cell_late_rate > 0)),
        }
    rankings = {}
    policies = [name for name in names if name != "baseline"]
    for i, first_name in enumerate(policies):
        for second_name in policies[i + 1:]:
            a, b = fan[names.index(first_name)], fan[names.index(second_name)]
            # Positive means the first scenario leaves less compaction than the second.
            cumulative_margin = b[:, ix[last]] - a[:, ix[last]]
            rate_margin = ((b[:, ix[last]] - b[:, ix[late_before]])
                           - (a[:, ix[last]] - a[:, ix[late_before]])) / (last - late_before)
            rankings[f"{first_name}_over_{second_name}"] = {
                f"cumulative_margin_{last}_cm": _summarize(cumulative_margin),
                "late_rate_margin_cm_per_year": _summarize(rate_margin),
            }
    return {
        "schema": 1, "task": "conditional-subsidence-scenario-diagnostic",
        "source_sha256": sha256(path), "scenarios": results, "pairwise_rankings": rankings,
        "monthly_forward_crosscheck": crosscheck,
        "policy_start_year": policy_start_year,
        "late_rate_years": [late_start_year, final_year],
        "comparison": "paired annual December subsidence; baseline minus policy",
        "rate_definition": "difference of December year-end cumulative compaction, cm/year",
        "ensemble_note": "member extrema and positive fractions are sensitivity diagnostics, not probabilities or predictive intervals",
        "field_validated_policy_effect": False, "intervention_feasibility_verified": False,
        "limitations": ["Only annual year-end states saved; monthly rebound timing is unresolved",
                        "Scenario inputs and future weather are assumed, not observed",
                        "Model classes disagree about sustained rate response",
                        "Compaction mechanism and spatial column boundary remain unvalidated"],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path, nargs="+",
                        help="saved yearly member sidecars for alternative physical models")
    parser.add_argument("--forward", type=Path, nargs="+",
                        help="optional monthly forward files, one per sidecar, for unit/year checks")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.forward is not None and len(args.forward) != len(args.sidecar):
        parser.error("--forward needs exactly one monthly file per --sidecar")
    forward = args.forward or [None] * len(args.sidecar)
    result = {path.stem: evaluate_sidecar(path, forward=monthly)
              for path, monthly in zip(args.sidecar, forward, strict=True)}
    atomic_json(args.out, result)
    for model, evidence in result.items():
        print(model)
        for name, values in evidence["scenarios"].items():
            final = evidence["late_rate_years"][1]
            start = evidence["late_rate_years"][0]
            end = values["year_end"][str(final)]["avoided_cumulative_cm"]["mean"]
            late = values["late_rate_reduction_cm_per_year"]["mean"]
            print(f"  {name}: {final} avoided {end:.2f} cm; {start}-{final} rate reduction "
                  f"{late:.3f} cm/year (conditional)")


if __name__ == "__main__":
    main()
