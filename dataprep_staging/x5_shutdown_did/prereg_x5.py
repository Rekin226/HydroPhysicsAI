"""Declare X5.shutdown_sdid.v1 (harness.prereg3, kind 'declaration') BEFORE any leveling or
ring value is read and before the synthetic check runs.

Route 5 of the 2026-10-25 audit: a model-free causal estimate of the subsidence response
(Q1) to the 2019 stop of the Yunlin Taiwan Water public-supply wells, by synthetic
difference-in-differences on annual leveling rates (and, reported only, ring bands).

Why a declaration and not a test: every leveling benchmark-year 2009-2021 has been seen in
design (the stage-3 column is calibrated on leveling to 2021; registry leveling folds are
design_time; S5 scored leveling 2012-2021, W3 2011-2021, N1 2007-2010 and 2016-2018; the
2026-10-25 audit computed per-benchmark rates 2009-2021 including the 2017-21 long
difference), and rings 2014-2021 were seen by S1a (design_time looks). No fresh unit exists for
this outcome, so the estimate is design-time evidence: its estimator, sets, windows, placebos
and thresholds are frozen here and it is computed once, but it never validates a model.

Run under the shared lock:
    flock results/dataprep/preregistration/.x5.lock python -I prereg_x5.py [--write]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import look_x5  # noqa: E402
import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
HERE = Path(__file__).resolve().parent
CODE = [HERE / f for f in ("x5const.py", "x5lib.py", "synth_check.py", "run_x5.py",
                           "look_x5.py", "prereg_x5.py")]
DATA = [K.WELLS_LOCATED, K.INTERVENTIONS, K.PUBLIC_WELL_JANMAY, K.ZONE_MAP, K.CENSUS_CELLS,
        K.RINGS, K.LEVELING_DATA_DIR / "ls_cache/ls-wra-lsp-obs__choushui_panel.parquet",
        K.LEVELING_DATA_DIR / "ls_cache/ls-wra-mlcw-obs__meta.parquet"]


def event_facts() -> dict:
    """Event description with every number read from the held tables at registration."""
    pw = pd.read_csv(K.PUBLIC_WELL_JANMAY, usecols=["county", "agency", "township", "year",
                                                    "value_1e4m3"])
    tw = pw[(pw.county == "Yunlin") & (pw.agency == K.TWC_AGENCY)]
    cty = tw[tw.township == "(county)"].set_index("year").value_1e4m3 / 100.0
    town = tw[tw.township != "(county)"].pivot(index="year", columns="township",
                                                values="value_1e4m3") / 100.0
    pre = cty.loc[2016:2018].mean()
    b02 = town.loc[2014:2018]
    w = pd.read_csv(K.WELLS_LOCATED, usecols=["county", "agency", "township"])
    w = w[(w.county == "Yunlin") & (w.agency == K.TWC_AGENCY)]
    u = L.unit_table()
    u = u[u.ongrid & u.date_balanced]
    tr, don = L.arm_sets(u, "T2")
    t1, _ = L.arm_sets(u, "T1")
    return {
        "what": "Yunlin Taiwan Water (台水) public-well Jan-May pumping (Mm3, chart reads, "
                "public_well_janmay.csv; rows B04YT/B05/B07 of interventions.csv): "
                + ", ".join(f"{y} {cty[y]:.2f}" for y in range(2011, 2023))
                + f"; 2016-18 mean {pre:.2f}. Hushan supply (M06). B02: no Jan-May TWC "
                "pumping in 虎尾/土庫/元長 since 2019.",
        "where_the_dose_is": (
            "TWC Jan-May pumping 2014-2018 in the B02 townships (Mm3): "
            + "; ".join(f"{c} {b02[c].min():.2f}-{b02[c].max():.2f}"
                        for c in ("虎尾", "土庫", "元長"))
            + f", i.e. {100 * b02[['虎尾', '土庫', '元長']].sum(1).mean() / pre:.1f} % of the "
            "county 2016-18 mean. The located proxy of the stopped wells is the "
            f"{len(w)} Yunlin TWC wells of the 2022 drought-well inventory (public_wells/v2 "
            "wells_located.csv): wells kept after 2019 and switched back on in 2021; "
            f"{int(w.township.isin(['虎尾鎮', '土庫鎮', '元長鄉']).sum())} lie in the B02 "
            f"townships, {int(w.township.isin(['斗六市', '古坑鄉']).sum())} in 斗六/古坑. "
            "Wells closed in 2017-18 (C03) are not located."),
        "timing_in_leveling": "interval y = May survey y-1 -> May survey y. 2019 holds "
                              "Jan-May 2019 off (about 5 of 12 months), 2020 is fully off, "
                              "2021 holds the Jan-May 2021 drought reactivation.",
        "date_only_counts": {"T2_treated": int(len(tr)), "T2_donors": int(len(don)),
                             "T2_donors_in_changhua": int(u.loc[don, "changhua"].sum()),
                             "T1_treated": int(len(t1)),
                             "note": "before the value filter (missing or |S| > 20 cm/yr)"},
    }


def doc() -> dict:
    return {
        "id": K.PREREG_ID, "wp": K.WP, "kind": "declaration",
        "title": "X5: synthetic DiD of the 2019 Yunlin Taiwan Water well stop on leveling "
                 "rates (design-time causal estimate of the Q1 subsidence response)",
        "tests": NA("declaration: design-time evidence; no fresh unit exists for leveling "
                    "2009-2021 or rings 2014-2021"),
        "label": "design-time evidence, computed once under this declaration; never "
                 "validation; cannot promote a model or a forcing",
        "event": event_facts(),
        "design": {
            "outcome": K.SIGN + "; one survey per benchmark-year (nearest 15 May); intervals "
                       f"of {K.DT_MIN}-{K.DT_MAX} yr; units: on-grid (<= 1 km from an active "
                       "cell), a survey every year 2008-2021, every interval admissible, no "
                       f"|S| > {K.GROSS_RATE} cm/yr (balanced panel, years 2009-2021)",
            "estimator": "synthetic DiD (Arkhangelsky et al. 2021, doi 10.1257/aer.20190159), "
                         "synthdid defaults (x5lib.sdid; Frank-Wolfe weights checked against "
                         "SLSQP on a random panel: max weight difference 5e-5)",
            "primary_arm_T2": f"treated: <= {K.R_TREAT:.0f} m from a proxy well; donors: "
                              f"> {K.R_BUFFER:.0f} m from every proxy well, outside the B02 "
                              f"townships and > {K.T1_DONOR_GAP:.0f} m from any B02 benchmark "
                              "(3-6 km annulus and B02 surroundings excluded). Donors are "
                              "mostly Changhua benchmarks (counts in event.date_only_counts): "
                              "an identification limit stated here",
            "periods": {"pre": K.PRE, "post_primary": K.POST, "post_alt_declared": K.POST_ALT},
            "inference": f"placebo-in-space by blocks: {K.N_PLACEBO} draws (seed {K.SEED}); "
                         f"each draw orders the {K.BLOCK_M:.0f} m donor blocks at random and "
                         "takes whole blocks until it holds >= N1 units; the rest of the "
                         "donors are its controls. p_one = (1 + #{tau_pl <= tau}) / (1 + B); "
                         "p_two = (1 + #{|tau_pl| >= |tau|}) / (1 + B); se = sd(tau_pl); "
                         "90 % interval tau +- 1.645 se",
            "lead_placebo": "fake event 2018: pre 2009-2017, post 2018, real post years "
                            "dropped; same sets and placebo scheme (seed + 1)",
            "alternatives_reported_not_gating": {
                "post_2019_2020": "drops the 2021 reactivation/drought interval",
                "per_year": "2019, 2020, 2021 separately",
                "radii": {k: list(v) for k, v in K.R_ALTS.items()},
                "period_pair_2009_13_vs_2015_19": "SDID with pre 2009-13 and post 2015-19 on "
                                                  "the T2 sets; only 2019 is post-event, so it "
                                                  "is a near-placebo (expected near 0 or at "
                                                  "most about a fifth of the primary ATT)",
                "arm_T1": "treated = benchmarks in 虎尾/土庫/元長 (B02 quote); donors as T2; "
                          "post 2019-21, 2019 only (before the 2020 Golden Corridor "
                          "extension), 2020-21, and the 2018 lead",
            },
            "golden_corridor_confounder": {
                "what": "M11/C01: crop-1 rice to dry crops within 1.5 km of the HSR (2013-20), "
                        "extended in 2020 to 虎尾/土庫/元長/北港; overlaps T1 entirely and lies "
                        "mostly in excluded areas for T2",
                "G1": "T2 with donors outside the four GC townships",
                "G2": "T2 on S - beta E, E = census kWh within 3 km over the interval, beta "
                      "from a two-way FE fit on the donors (Arkhangelsky covariate route); "
                      "the census does not meter the TWC public wells",
                "G3": "GC-only arm: treated = 北港 benchmarks, donors as T2, pre 2009-2019, "
                      "post 2020-21: the size of a GC-extension effect without a B02 stop",
                "G4": "T2 without treated units in GC townships",
                "reading": "if G1/G2/G4 keep the sign and G3 is near 0, the T2 estimate is not "
                           "the Golden Corridor; T1 cannot separate the two after 2019"},
            "rings_reported": "annual May-to-May band compaction rates (flag-0 primary series "
                              "of rings_bridged/v3, cm/yr, + = compaction), 2015-2018 pre, "
                              "2019-2021 post; treated = ring sites in the B02 townships (no "
                              "ring lies within 3 km of a proxy well); donors = the other "
                              "sites; leave-one-donor-out placebos (the smallest attainable p "
                              "is 1/(1 + donors), so rings cannot reach p < 0.10 and never "
                              "gate). Reported: ATT per band and deep (band2, 200 m-bottom) "
                              "minus shallow (band0, 0-100 m); 'deep responds more' = band2 "
                              "ATT below band0 ATT.",
        },
        "thresholds": {
            "success_primary": "ATT(T2, 3 km, 2019-21) < 0 AND one-sided block-placebo p < "
                               "0.10 AND lead placebo two-sided p >= 0.10 AND synthetic gate "
                               "passed; otherwise 'not supported' (or 'not evaluable' if the "
                               "synthetic gate failed). Deep-vs-shallow: reported.",
            "synthetic_gate": {
                "dgp": "see synth_check.py docstring (real survey dates and geometry, "
                       "trend + factor + 5 km block random walk + survey noise)",
                "panels": K.SYN_N_PANELS, "placebos": K.SYN_N_PLACEBO, "delta": K.SYN_DELTA,
                "A_bias": f"|mean tau - truth| <= {K.SYN_BIAS_TOL} |truth|",
                "A_coverage": f"placebo 90 % interval covers truth in {list(K.SYN_COVER)}",
                "A0_size": f"rate of one-sided p < 0.10 under no effect <= {K.SYN_SIZE_MAX}",
                "A0_lead": f"rate of lead two-sided p < 0.10 under no effect <= "
                           f"{K.SYN_SIZE_MAX}",
                "reported": "power at -0.5 and -0.25 cm/yr; scenario B (Yunlin-only shocks, "
                            f"sd {K.SYN_COUNTY_SD}) and C (effect decaying exp(-d/2 km) with "
                            "spillover to donors) bias and coverage"},
            "secondary_long_difference": {
                "what": "per benchmark, mean S over 2017-21 minus 2010-14 (>= 3 admissible "
                        "intervals each) against the change in mean census kWh within 3 km "
                        "over the same intervals, Yunlin, OLS on centred values, 5 km block "
                        f"bootstrap ({K.LD_BOOT}); the audit's -0.37 cm/yr per GWh/yr in the "
                        "elevation sign is +0.37 here (already seen: design-time)",
                "pass_if_all": ["slope > 0 with the 90 % interval above 0",
                                "alternative pair 2009-13 vs 2015-19: slope > 0",
                                "annual two-way FE (benchmark, year) of S on E(t) and the "
                                "lead E(t+1), Yunlin: the 90 % block-bootstrap interval of "
                                "the lead coefficient includes 0"],
                "reported": "the long difference with a T2 indicator (TWC wells are not in the "
                            "census), Changhua's slope"},
            "attempt_policy": "engineering retries (crash, NaN, missing file) before any metric "
                              "is seen are free and logged; run_x5.py writes report.json once; "
                              "at most one exploratory follow-up after a failure, labelled "
                              "'post-failure, exploratory; never validation; cannot promote'",
            "u7": "no 2023+ value is read (leveling ends 2021-09, rings cut at 2021-12, census "
                  "energy to 2022-05 for the lead term only)",
        },
        "external_units": look_x5.members(),
        "prior_looks_disclosure": {
            "registry_logged": "S5 score leveling:2012-01..2021-12, 2018-01..2019-12, "
                               "2020-01..2021-12; W3 score leveling 2011-2021; N1 score "
                               "leveling 2007-2010 and 2016-2018; I1 history heads+leveling "
                               "2012-2022; S1a design_time rings+piles 2014-2025",
            "unlogged": "the 2026-10-25 audit (lit-causal agent) computed per-benchmark "
                        "interval rates 2009-2021 against census kWh (TWFE and the 2017-21 vs "
                        "2010-14 long difference) and ring band rates 2010-21 by site; the "
                        "stage-3 column is calibrated on leveling to 2021",
            "this_wp": "survey dates, locations, townships and distances only (logged as "
                       "design_availability before this declaration); a 2-panel, 20-placebo "
                       "synthetic dry run (engineering; seen: gate outcome on 2 panels, "
                       "nothing changed after it); a smoke run of run_x5.py on random numbers"},
        "zones": NA("no model zones; benchmark geometry only"),
        "priors": NA("model-free"), "burn_in": NA("none"), "folds": NA("none"),
        "fresh_units": NA("no fresh units: leveling 2009-2021 and rings 2014-2021 were seen "
                          "in design"),
        "gate_scores": NA("no model gate"),
        "primary_model": NA("model-free estimator; the twin rung it may inform is stage 2"),
        "inputs": [rel_to_repo(p) for p in CODE + DATA],
        "interpretation": (
            "A supported primary says: benchmarks within 3 km of the stopped Taiwan Water "
            "wells sank less after 2019 than a synthetic control of distant benchmarks, beyond "
            "what block placebos produce, with no such gap in 2018. It sizes a 1-3-year "
            "response to a known deep-pumping cut (per Mm3 via the B05 step), as design-time "
            "evidence and a target for a twin rung, not as validation. 'Not supported' says "
            "the leveling cannot separate a response of that size from block-level noise with "
            "this donor pool; it does not say the response is zero."),
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    if errs or not a.write:
        return 1 if errs else 0
    path, sha = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
