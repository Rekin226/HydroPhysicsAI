"""Controlled vocabulary of product uses and the use rules U1-U4/U7 (plan 2.5, 4.1, 4.2).

A manifest's ``allowed_uses`` / ``forbidden_uses`` and a loader's declared use must come from
:data:`USES`. ``PRE_ORIGIN_USES`` are the uses that calibrate, start, set a prior for, design or
select a model scored on 2023-26 (U3, U7): they need ``fit_window_end <= ORIGIN`` unless a
listed exemption covers them (U5) or the consumer reads a time slice ending by the origin
(``guard.open_product(..., window_end=...)`` on a product that declares ``time_slicing``).

Repair round (2026-10-12): ``datum_fit`` and ``baseline`` are fitted uses (a per-well datum
is a calibrated offset of the scored model; a climatology or persistence baseline is a model
fitted to observations and compared with the score), so they are pre-origin uses too.
``design_time_evidence`` informs design choices of models scored on 2023-26 (plan 4.5), so
it is a pre-origin use as well: pilot or design evidence fitted on post-origin data is
available only as ``after_the_fact_analysis``. A reference statistic computed from the test
period itself (for example the denominator of an NSE) is part of the score: compute it under
``scoring_target``.
"""
from __future__ import annotations

USES = {
    # U1: observed values only (flags 0-1)
    "scoring_target": "target of a reported score (2023-26 test, folds, held-out sites)",
    "baseline": "climatology or other baseline a score is compared with",
    "datum_fit": "per-well datum or offset fit",
    "prospective_scorer": "the registered prospective scorer (Nov 2026 - Oct 2027)",
    # U2: observed plus daily-bridged months (flags 0-2), weighted by sigma_obs
    "calibration_target": "target of a model calibration (heads, leveling, rings)",
    # pre-origin model inputs (U3)
    "initial_conditions": "initial head field or state of a model run",
    "idw_field": "IDW or similar head field for a model start or an observation operator",
    "prior": "prior mean for a model parameter or forcing",
    "prior_width_input": "input to a prior's width (sd), not its level",
    "model_design": "zone design, burn-in, basis or other design choice of a model",
    "selection": "gate, ladder decision, member choice, hyperparameter or threshold (U7)",
    "forcing_frozen": "forcing of the frozen or registered prospective forecasts",
    # post-origin diagnostic and analysis uses
    "forcing_recorded_arm": "forcing of the recorded-forcing diagnostic arm only (U4)",
    "validator": "independent check of a reported number (one role per number, 4.1)",
    "statistical_layer_history": "history fed to the statistical layer (full variants)",
    "after_the_fact_analysis": "after-the-fact analysis, never fed back into a scored model",
    "design_time_evidence": "pilot or design-time evidence; never validation of its own choices",
}

#: U1 - flags 2-6 never enter these.
OBSERVED_ONLY_USES = frozenset({"scoring_target", "baseline", "datum_fit", "prospective_scorer"})

#: U3/U7 - require fit_window_end <= ORIGIN (or a covering exemption, or a time slice).
PRE_ORIGIN_USES = frozenset({
    "calibration_target", "datum_fit", "baseline", "initial_conditions", "idw_field", "prior",
    "prior_width_input", "model_design", "selection", "forcing_frozen", "design_time_evidence",
})

#: Uses an exemption (U3: an input that cannot be had pre-origin, such as the LUIMAP 2024
#: snapshot) may cover. Targets and fits to observations (calibration_target, datum_fit,
#: baseline) and selection (U7 has no exemption clause) can never be exempted.
EXEMPTABLE_USES = frozenset({
    "initial_conditions", "idw_field", "prior", "prior_width_input", "model_design",
    "forcing_frozen", "design_time_evidence",
})

#: Uses that may read a product whose validation gate has not passed.
GATE_EXEMPT_USES = frozenset({"design_time_evidence", "after_the_fact_analysis"})

#: Allowed row flags per use (U1, U2; flag 6 never enters calibration; 8 and 9 never by
#: default). Uses not listed get DEFAULT_FLAGS.
FLAGS_BY_USE = {
    "scoring_target": frozenset({0, 1}),
    "baseline": frozenset({0, 1}),
    "datum_fit": frozenset({0, 1}),
    "prospective_scorer": frozenset({0, 1}),
    "calibration_target": frozenset({0, 1, 2}),
    "initial_conditions": frozenset({0, 1, 2, 3, 4}),
    "idw_field": frozenset({0, 1, 2, 3, 4}),
    "prior": frozenset({0, 1, 2, 3, 4, 5}),
    "prior_width_input": frozenset({0, 1, 2, 3, 4, 5}),
    "model_design": frozenset({0, 1, 2, 3, 4, 5}),
    "selection": frozenset({0, 1, 2, 3, 4, 5}),
    "forcing_frozen": frozenset({0, 1, 2, 3, 4, 5}),
}
DEFAULT_FLAGS = frozenset({0, 1, 2, 3, 4, 5, 6})

#: Ledger roles (plan 4.1); stored in snake_case.
ROLES = ("fit_target", "prior", "prior_width_input", "input", "validator",
         "design_time_evidence")


def normalise_role(role: str) -> str:
    r = role.strip().lower().replace("-", "_").replace(" ", "_")
    if r not in ROLES:
        raise ValueError(f"unknown ledger role {role!r}; expected one of {ROLES}")
    return r


def flags_for_use(use: str) -> frozenset:
    if use not in USES:
        raise ValueError(f"unknown use {use!r}; expected one of {sorted(USES)}")
    return FLAGS_BY_USE.get(use, DEFAULT_FLAGS)
