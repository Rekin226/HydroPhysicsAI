"""Synthetic Tuku depth operator and design-identifiability diagnostic.

Every compliance and head path here is invented for a numerical experiment.
No ring observation, head measurement or site material parameter is fitted.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from hydrophysics.twin.site_geometry import TukuGeometry, load_tuku_geometry
from hydrophysics.twin.tuku_sensitivity import ADJACENT_HEAD_DRIVER

SYNTHETIC_COMPLIANCE = 2e-4  # m shortening / (m interval * m head drawdown)
SYNTHETIC_NOISE_M = 0.001  # illustrative monthly ring-change noise, not site QC
RELATIVE_95_TOLERANCE = 0.5  # explicit design target, not a calibrated criterion


@dataclass(frozen=True)
class Atom:
    top: float
    bottom: float
    parameter: int
    head_driver: np.ndarray


@dataclass(frozen=True)
class Design:
    name: str
    parameter_names: tuple[str, ...]
    atoms: tuple[Atom, ...]


def designs(g: TukuGeometry) -> dict[str, Design]:
    """Three explicit structural hypotheses; IOT/WRA stress mapping is assumed."""
    driver = ADJACENT_HEAD_DRIVER
    hydro = g.hydro_bounds_m_bgs
    confining = (1, 3, 5)
    wra = Design(
        "WRA_3_confining_coefficients",
        tuple(g.hydro_names[j] for j in confining),
        tuple(Atom(*hydro[j], i, driver[j]) for i, j in enumerate(confining)),
    )
    fine = [(i, lo, hi) for i, ((lo, hi), code)
            in enumerate(zip(g.soil_bounds_m_bgs, g.soil_classes, strict=True))
            if code in {"CL", "ML"}]
    fine_atoms = []
    zone_atoms = []
    for _p, lo, hi in fine:
        # One IOT soil interval gets one head driver: thickness-weighted from
        # the independently dated WRA hydro interpretation. This is unverified.
        mapped = np.zeros(4)
        for j, (hlo, hhi) in enumerate(hydro):
            length = max(0.0, min(hi, hhi) - max(lo, hlo))
            mapped += length * driver[j]
            if length:
                zone_atoms.append(Atom(max(lo, hlo), min(hi, hhi), j, driver[j]))
        fine_atoms.append(Atom(lo, hi, len(fine_atoms), mapped / (hi - lo)))
    iot_fine = Design("IOT_29_fine_interval_coefficients",
                      tuple(f"soil_{p + 1}_{g.soil_classes[p]}" for p, _, _ in fine),
                      tuple(fine_atoms))
    iot_zones = Design("IOT_fine_support_7_hydro_zone_coefficients",
                       tuple(g.hydro_names), tuple(zone_atoms))
    return {x.name: x for x in (wra, iot_fine, iot_zones)}


def ring_depths(g: TukuGeometry) -> np.ndarray:
    """Original installed NO2–NO24 only; NO1 and deeper NO25/26 are unresolved."""
    depths = g.installed_ring_depths_m_bgs[1:24]
    if len(depths) != 23 or depths[-1] > 300 or np.any(np.diff(depths) <= 0):
        raise ValueError("Expected ordered installed NO2–NO24 inside logged depth")
    return depths


def matrix(design: Design, heads_drawdown_m: np.ndarray,
           depths_m_bgs: np.ndarray) -> np.ndarray:
    """Rows are time × adjacent-ring intervals; columns are compliance terms."""
    h = np.asarray(heads_drawdown_m, dtype=float)
    depths = np.asarray(depths_m_bgs, dtype=float)
    if (h.ndim != 2 or h.shape[1] != 4 or not np.isfinite(h).all()
            or depths.ndim != 1 or len(depths) < 2
            or np.any(np.diff(depths) <= 0) or not np.isfinite(depths).all()):
        raise ValueError("Expected finite four-head path and ordered ring depths")
    r = len(depths) - 1
    x = np.zeros((len(h), r, len(design.parameter_names)))
    for atom in design.atoms:
        overlap = np.maximum(0.0, np.minimum(depths[1:], atom.bottom)
                             - np.maximum(depths[:-1], atom.top))
        x[:, :, atom.parameter] += np.outer(h @ atom.head_driver, overlap)
    return x.reshape(len(h) * r, -1)


def cumulative(design: Design, heads_drawdown_m: np.ndarray,
               compliance: np.ndarray, depths_m_bgs: np.ndarray) -> np.ndarray:
    """Independent cumulative-depth construction for ring integration checks."""
    h = np.asarray(heads_drawdown_m, dtype=float)
    k = np.asarray(compliance, dtype=float)
    depth = np.asarray(depths_m_bgs, dtype=float)
    if k.shape != (len(design.parameter_names),):
        raise ValueError("Compliance vector does not match design")
    out = np.zeros((len(h), len(depth)))
    for atom in design.atoms:
        thickness = np.clip(depth - atom.top, 0.0, atom.bottom - atom.top)
        out += np.outer(h @ atom.head_driver, thickness) * k[atom.parameter]
    return out


def synthetic_heads() -> np.ndarray:
    """Correlated but distinct four-layer drawdown paths, in metres."""
    t = np.arange(12, dtype=float)
    return np.stack([0.25 * (1 + 0.1 * j) * t
                     + 0.3 * (np.sin(2 * np.pi * t / 12 + 0.6 * j)
                              - np.sin(0.6 * j)) for j in range(4)], axis=1)


def _rank_and_condition(x: np.ndarray) -> tuple[int, float | None, np.ndarray, np.ndarray]:
    _, values, right = np.linalg.svd(x, full_matrices=True)
    rank = int(np.count_nonzero(values > values[0] * 1e-10))
    cond = float(values[0] / values[-1]) if rank == x.shape[1] else None
    return rank, cond, values, right


def analyze(sigma_m: float = SYNTHETIC_NOISE_M,
            relative_95_tolerance: float = RELATIVE_95_TOLERANCE) -> dict:
    if not (np.isfinite(sigma_m) and sigma_m > 0
            and np.isfinite(relative_95_tolerance) and relative_95_tolerance > 0):
        raise ValueError("Positive finite synthetic noise and tolerance required")
    g = load_tuku_geometry()
    depth = ring_depths(g)
    h_basis = np.eye(4)  # maximum independent information from four drivers
    h_practical = synthetic_heads()
    result = {}
    for name, design in designs(g).items():
        n = len(design.parameter_names)
        x_basis = matrix(design, h_basis, depth)
        rank, basis_cond, _, _ = _rank_and_condition(x_basis)
        x = matrix(design, h_practical, depth)
        p_rank, p_cond, values, right = _rank_and_condition(x)
        k = np.full(n, SYNTHETIC_COMPLIANCE)
        ring = (x @ k).reshape(len(h_practical), -1)
        cumulative_ring = np.diff(cumulative(design, h_practical, k, depth), axis=1)
        ring_error = float(np.max(np.abs(ring - cumulative_ring)))
        whole = cumulative(design, h_practical, k, np.array([0.0, 300.0]))
        whole_from_atoms = matrix(design, h_practical, np.array([0.0, 300.0])) @ k
        whole_error = float(np.max(np.abs(whole[:, 1] - whole_from_atoms)))
        if rank == n and p_rank == n:
            recovered = np.linalg.lstsq(x, x @ k, rcond=None)[0]
            cov_diag = np.sum((right.T / values) ** 2, axis=1) * sigma_m ** 2
            worst_relative_95 = float(1.96 * np.sqrt(cov_diag).max()
                                      / SYNTHETIC_COMPLIANCE)
            gate = ("passes_synthetic_design_only" if worst_relative_95
                    <= relative_95_tolerance else "reject_practical_at_declared_noise")
            recovery_error = float(np.max(np.abs(recovered - k)))
            indistinguishable_pair = None
        else:
            gate = "reject_structural_rank_deficiency"
            worst_relative_95 = None
            recovery_error = None
            null_direction = right[rank]
            other = k + (0.5 * SYNTHETIC_COMPLIANCE
                         / np.max(np.abs(null_direction))) * null_direction
            indistinguishable_pair = {
                "min_alternative_positive_compliance": float(other.min()),
                "max_coefficient_difference": float(np.max(np.abs(other - k))),
                "max_ring_prediction_difference_m": float(np.max(np.abs(x @ other - x @ k))),
            }
        result[name] = {
            "n_parameters": n, "basis_rank": rank, "practical_rank": p_rank,
            "nullity": n - rank, "basis_condition": basis_cond,
            "practical_condition": p_cond,
            "max_relative_95_halfwidth_at_synthetic_noise": worst_relative_95,
            "noiseless_recovery_max_abs": recovery_error,
            "ring_cumulative_identity_error_m": ring_error,
            "whole_column_identity_error_m": whole_error,
            "final_whole_column_synthetic_m": float(whole[-1, -1]),
            "gate": gate, "positive_indistinguishable_pair": indistinguishable_pair,
        }
    return {
        "status": "SYNTHETIC_NUMERICAL_DESIGN_ONLY_NO_SITE_FIT",
        "ring_ids": "installed NO2–NO24", "n_adjacent_ring_intervals": len(depth) - 1,
        "n_synthetic_months": len(h_practical),
        "synthetic_compliance_m_per_m_per_m": SYNTHETIC_COMPLIANCE,
        "synthetic_ring_noise_m": sigma_m,
        "relative_95_halfwidth_limit": relative_95_tolerance,
        "assumptions": [
            "WRA hydro units and IOT fine soil supports are separate structural hypotheses",
            "IOT head drivers use WRA hydro contacts, an unverified cross-source mapping",
            "No consolidation law, drainage time, station pressure, or field ring values are fitted",
            "Practical interval uses independent 1 mm synthetic errors as an illustrative design target",
        ],
        "designs": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-mm", type=float, default=SYNTHETIC_NOISE_M * 1000)
    parser.add_argument("--relative-95-tolerance", type=float,
                        default=RELATIVE_95_TOLERANCE)
    args = parser.parse_args()
    print(json.dumps(analyze(args.noise_mm / 1000, args.relative_95_tolerance),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
