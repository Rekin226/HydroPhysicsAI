"""Validation and anti-leakage harness (plan WP A1, sections 4.3-4.7).

Shared code so that every work package scores, gates, placebos, masks and pre-registers the
same way:

  scorers    equal-well, anomaly and out-of-time RMSE (training-mean datum), native and
             split-conformal 90 % coverage, pseudo-truth recovery metrics, ratios of totals
  coverage   cluster-adjusted Wilson interval for a coverage rate and the 4.4 gate rule
  placebos   6-month shift, own climatology, year-shuffled profile, distant-kWh and regional
             covariates, rain-deficit index
  gaps       artificial-gap and injection generators for every family of plan 4.4, with the
             fixed seeds ``1000*ord(family)+r`` (``7000+r`` for the test-period replica)
  units      design-time and fresh validation units, shaped stress regions, site-grouped and
             spatial-block folds with buffers
  prereg     hashed pre-registrations, the look log and the refusal to score a design whose
             pre-registration postdates a look at its test data

Import with ``sys.path.insert(0, "<repo>/dataprep_staging")`` and ``from harness import
scorers``. Nothing under ``hydrophysics/`` imports this package; it imports repository
modules read-only where needed.
"""
from __future__ import annotations

__version__ = "1.0.0"   # WP A1, 2026-10-12
