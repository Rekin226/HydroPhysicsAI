"""N2CS OSSE engine: the synthetic decidability gate (G0) and the form choice, model-only.

Order (the lead runs it; nothing here runs on import):

1. ``extract``: from the two CPU forwards of R1c's main set (run_n2cs.sh forward-a / forward-b;
   N2C's registered command with the arms replaced) it checks the forward metadata and the
   recorded-pumping file hashes, applies the PRE-GATE (both unit arms reproduce the run's
   baseline (the ZR arm) within 1e-6 m at every layer, cell and month, and the two runs'
   baselines agree within 1e-6 m), and writes the contrast panel: c_CH (7, 74, 11) and c_YL
   (7, 74, 11) at N2C's 74 Changhua screens over 2026-12..2027-10, the replays' window
   anomalies and the 5 km blocks. A failed pre-gate is an ENGINEERING failure: no panel.
0. ``declare --seed S``: ``draft_osse_declaration(S)`` (harness.prereg3 kind 'declaration',
   N2CS.osse_design.v1: code shas, the world seed, G0, the choice rule, the forms, the maximum
   filled fraction; only files that exist) is validated here; the LEAD registers it BEFORE
   ``run_n2cs.sh files`` (X5.n2c_reading precedent). Nothing here writes a registration.
2. ``run``: refuses unless the declaration is registered, the code is unchanged since it and
   the panel is the one whose sha256 ``extract`` appended to attempts.jsonl; then runs the
   declared OSSE once (n2cs_lib.run_osse) and writes result.json once.

``extract`` also refuses if the code changed since the declaration, checks every arm file's
per-cell path (one county per arm, 1 elsewhere) and appends the panel sha256 to attempts.jsonl.

No observation is read anywhere: worlds are e = k c_CH + k c_YL + noise + confounder with the
published noise parameters of synth_n2c.PARAMS (design-time values cited there), and the
forcing variability is the pre-origin census (pump_electricity/preorigin/v3, use
design_time_evidence). No rain covariate.

    python -I dataprep_staging/n2cs/osse_n2cs.py declare --seed <int>   (validate only)
    python dataprep_staging/n2cs/osse_n2cs.py extract
    python dataprep_staging/n2cs/osse_n2cs.py run
    python dataprep_staging/n2cs/osse_n2cs.py smoke --seed <int> --out <dir>   (synthetic)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "n2_policy"),
           str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import n2cs_const as K  # noqa: E402
import n2cs_lib as L  # noqa: E402

DECL_ID = K.OSSE_DECL_ID
CODE = (f"{K.X}/n2cs_const.py", f"{K.X}/n2cs_lib.py", f"{K.X}/osse_n2cs.py",
        f"{K.X}/build_r_county.py", f"{K.X}/run_n2cs.sh", K.SCORER, K.SYNTH, K.REPLAY)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(s) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_attempt(rec: dict, path: Path | None = None) -> None:
    p = path or rp(K.ATTEMPTS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec, time=utc()), sort_keys=True) + "\n")


def code_sha256() -> dict:
    return {p: sha256(rp(p)) if rp(p).exists() else "missing" for p in CODE}


def noise_params() -> dict:
    """The declared noise and confounder values, read from synth_n2c.PARAMS (bound sha)."""
    if sha256(rp(K.SYNTH)) != K.SYNTH_SHA:
        raise SystemExit(f"{K.SYNTH} differs from its bound sha {K.SYNTH_SHA[:8]}")
    sys.path.insert(0, str(REPO / "dataprep_staging" / "n2c_reading"))
    import synth_n2c as S

    return {k: float(S.PARAMS[k]["value"]) for k in ("sigma_tot_m", "block_share", "phi",
                                                       "kappa_m")}


# ----------------------------------------------------------------------------- extract
def screens() -> dict:
    p = rp(K.WELLS)
    if sha256(p) != K.WELLS_SHA:
        raise SystemExit(f"{K.WELLS} differs from {K.WELLS_SHA[:8]}")
    rows = [r for r in json.loads(p.read_text())["rows"] if r["county"] == "changhua"]
    if len(rows) != K.N_WELLS:
        raise SystemExit(f"expected {K.N_WELLS} Changhua screens")
    return {"sid": np.array([r["sid"] for r in rows]),
            "cell": np.array([r["cell"] for r in rows], int),
            "layer": np.array([r["layer"] for r in rows], int)}


def blocks_for(sids) -> np.ndarray:
    import pandas as pd

    t = pd.read_csv(rp(K.WELLS_XY), usecols=["sid", "x", "y"], dtype={"sid": str})
    t = t.drop_duplicates("sid").set_index("sid")
    miss = [s for s in sids if s not in t.index]
    if miss:
        raise SystemExit(f"{K.WELLS_XY}: no coordinates for {miss[:5]}")
    return BS.block_ids(t.loc[list(sids), "x"].to_numpy(float),
                        t.loc[list(sids), "y"].to_numpy(float))


def forward_meta_problems(f, names_expected: list[str]) -> list[str]:
    e = []
    names = [str(s) for s in f["scenario_names"]]
    if names != names_expected:
        e.append(f"scenarios {names} != {names_expected}")
    dates = [str(s)[:7] for s in f["dates"]]
    if len(dates) != K.N_T or dates[0] != "2012-01" or dates[-1] != "2032-12":
        e.append(f"dates {dates[:1]}..{dates[-1:]}")
    elif dates[K.FWD_WINDOW[0]] != K.WINDOW[0] or dates[K.FWD_WINDOW[1] - 1] != K.WINDOW[1]:
        e.append("window months misplaced")
    if int(f["origin"]) != K.ORIGIN or int(f["horizon"]) != K.HORIZON:
        e.append(f"origin {int(f['origin'])} horizon {int(f['horizon'])}")
    if [str(s) for s in f["member_labels"]] != [K.FWD_MEMBER]:
        e.append(f"member_labels {[str(s) for s in f['member_labels']]}")
    if float(f["gain"]) != 1.0 or float(f["hindcast_gain"]) != 0.0:
        e.append(f"gain {float(f['gain'])} hindcast_gain {float(f['hindcast_gain'])}")
    fo = json.loads(str(f["forward_options"]))
    bad = {k: fo.get(k) for k, v in K.FWD_OPTIONS.items() if fo.get(k) != v}
    if bad:
        e.append(f"forward_options differ: {bad}")
    return e


def arm_sha_problems(f, arms: dict) -> list[str]:
    """Each recorded arm was run on the file as built (sha256 in the forward's record)."""
    rec = json.loads(str(f["forward_options"])).get("recorded_pumping") or {}
    return [f"{a} was not run on {p}" for a, p in arms.items()
            if rec.get(a, {}).get("sha256") != sha256(rp(p))]


def pregate(hm: np.ndarray, names: list[str], arms=K.PREGATE_ARMS) -> dict:
    """Both unit arms against the baseline of the SAME run, every layer, cell and month."""
    base = np.asarray(hm[names.index("baseline")], float)
    out = {"tol_m": K.PREGATE_TOL_M}
    for a in arms:
        d = np.asarray(hm[names.index(a)], float) - base
        out[a] = {"max_abs_m": float(np.nanmax(np.abs(d))),
                  "finite": bool(np.isfinite(d).all())}
    out["ok"] = all(out[a]["finite"] and out[a]["max_abs_m"] <= K.PREGATE_TOL_M for a in arms)
    return out


def declared_code_problems() -> list[str]:
    """The registered OSSE declaration exists and the bound code is unchanged since it."""
    try:
        from harness import prereg3

        doc, _ = prereg3.load_prereg(DECL_ID)
    except Exception as exc:                       # noqa: BLE001 - any failure refuses
        return [f"{DECL_ID} is not registered ({type(exc).__name__})"]
    now = code_sha256()
    return [f"{k} changed since {DECL_ID}" for k, v in doc["code_sha256"].items()
            if now.get(k) != v]


def arm_file_problems(arms: dict, scopes: dict) -> list[str]:
    """Per-cell path (lead ruling S7): each arm file moves one county only (multiplier 1 on
    every cell of the other county), on the county map R1c was calibrated on."""
    import build_r_county as BR

    flags, _ = BR.load_flags(rp(K.COUNTY_MAP))
    out = []
    for a, path in arms.items():
        out += [f"{a}: {x}" for x in BR.arm_problems(rp(path), flags, scopes[a])]
    return out


def extract(out: Path = None) -> dict:
    bad = declared_code_problems()
    if bad:
        raise SystemExit("REFUSED: " + "; ".join(bad))
    out = rp(out or K.PANEL)
    rep_dir = rp(K.REPLAYS)
    rj = json.loads((rep_dir / "replays.json").read_text())
    names_a = ["baseline", *K.PREGATE_ARMS, *[f"RCH{y}" for y in K.REPLAY_YEARS]]
    names_b = ["baseline", *[f"RYL{y}" for y in K.REPLAY_YEARS]]
    arms_a = {"UCH": K.UNIT_CH, "UYL": K.UNIT_YL,
              **{f"RCH{y}": f"{K.REPLAYS}/rch{y}.npz" for y in K.REPLAY_YEARS}}
    arms_b = {f"RYL{y}": f"{K.REPLAYS}/ryl{y}.npz" for y in K.REPLAY_YEARS}
    fa = np.load(rp(K.FWD_A + ".npz"), allow_pickle=False)
    fb = np.load(rp(K.FWD_B + ".npz"), allow_pickle=False)
    scopes = {a: ("changhua" if a in ("UCH",) or a.startswith("RCH") else "yunlin")
              for a in (*arms_a, *arms_b)}
    probs = (forward_meta_problems(fa, names_a) + forward_meta_problems(fb, names_b)
             + arm_sha_problems(fa, arms_a) + arm_sha_problems(fb, arms_b)
             + arm_file_problems({**arms_a, **arms_b}, scopes))
    if probs:
        raise SystemExit("ENGINEERING: " + "; ".join(probs))
    hma, hmb = fa["heads_mean"], fb["heads_mean"]
    pg = pregate(hma, names_a)
    base_ab = float(np.nanmax(np.abs(np.asarray(hma[0], float) - np.asarray(hmb[0], float))))
    pg["baseline_a_vs_b_max_abs_m"] = base_ab
    pg["ok"] = bool(pg["ok"] and base_ab <= K.PREGATE_TOL_M)
    rec = {"pregate": pg, "forward_a_sha256": sha256(rp(K.FWD_A + ".npz")),
           "forward_b_sha256": sha256(rp(K.FWD_B + ".npz")),
           "replays_json_sha256": sha256(rep_dir / "replays.json")}
    if not pg["ok"]:
        log_attempt({"event": "extract", "pregate_ok": False, "pregate": pg})
        raise SystemExit(f"ENGINEERING: pre-gate failed {json.dumps(pg)}")
    sc = screens()
    t0, t1 = K.FWD_WINDOW
    ba = np.asarray(hma[0][sc["layer"], sc["cell"], t0:t1], float)
    bb = np.asarray(hmb[0][sc["layer"], sc["cell"], t0:t1], float)
    c_ch, c_yl, a_ch, a_yl = [], [], [], []
    for y in K.REPLAY_YEARS:
        c_ch.append(np.asarray(hma[names_a.index(f"RCH{y}")][sc["layer"], sc["cell"], t0:t1],
                               float) - ba)
        c_yl.append(np.asarray(hmb[names_b.index(f"RYL{y}")][sc["layer"], sc["cell"], t0:t1],
                               float) - bb)
        w = rj["replays"][f"rp{y}"]["window"]
        if w["months"] != L.WINDOW_MONTHS:
            raise SystemExit("replays.json window months differ")
        a_ch.append(w["a_changhua"])
        a_yl.append(w["a_yunlin"])
    panel = {"c_ch": np.stack(c_ch), "c_yl": np.stack(c_yl), "a_ch": np.array(a_ch, float),
             "a_yl": np.array(a_yl, float), "sid": sc["sid"], "block": blocks_for(sc["sid"]),
             "months": np.array(L.WINDOW_MONTHS)}
    if not (np.isfinite(panel["c_ch"]).all() and np.isfinite(panel["c_yl"]).all()):
        raise SystemExit("ENGINEERING: non-finite contrast")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        np.savez(fh, **panel)
    rec.update({"panel": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
                "panel_sha256": sha256(out), "n_blocks": int(len(set(panel["block"]))),
                "wells_sha256": K.WELLS_SHA, "wells_xy_sha256": sha256(rp(K.WELLS_XY))})
    with rp(K.PANEL_RECORD).open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    log_attempt({"event": "extract", "pregate_ok": True, "panel_sha256": rec["panel_sha256"],
                 "panel_record_sha256": sha256(rp(K.PANEL_RECORD))})
    return rec


def load_panel(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- design
def draft_osse_record(seed: int, *, params: dict | None = None) -> dict:
    """The OSSE design (the declaration's ``design`` block): spec, forms, choice rule, seeds.
    ``seed`` (the world seed) is required; the bootstrap seed is the pinned scorer's."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int (no default)")
    params = params if params is not None else noise_params()
    return {
        "estimand": K.ESTIMAND, "window": list(K.WINDOW), "n_wells": K.N_WELLS,
        "world": "e = k c_CH,j + k_YL c_YL,j + block AR(1) + well AR(1) + kappa load_i g_t over "
                 "N2C's 74 Changhua screens x 11 window months (ZR = 0, obs = e); c_CH,j / "
                 "c_YL,j = R1c's one-county contrasts of replay j (CPU forwards A / B, "
                 "baseline of the same run subtracted); k_YL = k unless a report cell says "
                 "otherwise; no rain covariate (no held product grants it a scoring role)",
        "params": params, "params_source": {"path": K.SYNTH, "sha256": K.SYNTH_SHA,
                                            "field": "PARAMS (sigma_tot_m, block_share, phi, "
                                                     "kappa_m) with their cited reports"},
        "forcing_variability": {"product": K.V3, "use": K.V3_USE, "table": K.V3_TABLE,
                                "replays": [f"rp{y}" for y in K.REPLAY_YEARS],
                                "max_filled_fraction": K.MAX_FILLED_FRACTION,
                                "note": "pre-origin census only (not electricity_census/v1); "
                                        "X1 forbids design_time_evidence"},
        "k_grid": list(K.K_GRID), "confounders": K.CONFOUNDERS,
        "gate_variants": K.GATE_VARIANTS, "report_cells": list(K.REPORT_CELLS),
        "replicates_per_cell": K.REPLICATES,
        "forms": {f: K.FORM_TEXT[f] for f in K.FORMS}, "interval": K.INTERVAL,
        "min_well_months": K.MIN_WELL_MONTHS,
        "readings": K.READING_TEXT, "size_x": K.SIZE_X,
        "informative_gate": {"form": "the form being scored (at scoring: the chosen form)",
                             "rms_of": "c_CH~ residualised on c_YL~ (Frisch-Waugh)",
                             "rms_m": K.INFORMATIVE_RMS_M, "max_abs_corr": K.MAX_ABS_CORR,
                             "min_clusters": BS.MIN_CLUSTERS, "min_months": BS.MIN_MONTHS,
                             "in_osse": "per replay and form, from its contrast (all 11 "
                                        "months recorded); a world on a non-informative "
                                        "replay reads not_informative"},
        "g0": K.G0, "g0_rule": "k = 1 and k = 1/3 only; every condition under every gate "
                               "variant; P(not overstated | k = 1) unconditional; the two false "
                               "rates the stricter of unconditional and given-informative; "
                               "the county-uniform variant is absorbed exactly by the month "
                               "effects of every form (by construction), so the binding "
                               "confounder variants are the shaped ones",
        "choice_rule": K.CHOICE_RULE,
        "plumbing": f"no noise, no confounder: beta_CH == k_CH within {K.PLUMBING_TOL} for "
                    "every replay and form; failure = engineering",
        "pregate": f"both unit arms reproduce the run's baseline within {K.PREGATE_TOL_M} m "
                   "(every layer, cell, month); forwards A and B agree; every arm file moves "
                   "one county only",
        "seeds": {"world_seed": seed, "bootstrap_seed": BS.SEED, "bootstrap_B": BS.B_BOOT},
        "seeding": K.SEEDING,
        "observations_read": "none (model output, published design-time parameters and the "
                             "pre-origin census only)"}


SEALING = ("the forward's stdout and outputs and the panel are not opened, printed or "
           "summarised before result.json exists (the runner prints shas; extract prints the "
           "panel shape, block count and the reproduction check only)")
DESIGN_DEVICE = ("the replays use twin.forward's recorded-pumping mechanism for synthetic "
                 "future months under the census product's allowed use design_time_evidence "
                 "(its MANIFEST forbids forcing_recorded_arm, which is not the use made here)")
DECL_INPUTS = (K.WELLS, K.WELLS_XY, K.COUNTY_MAP, f"{K.V3}/MANIFEST.json", K.RULINGS,
               f"results/dataprep/preregistration/{K.N2C_ID}.json", K.R1C_THETA, K.R1C_COLUMN)


def forward_commands() -> dict:
    flags = K.FWD_FLAGS
    base = (f"results/twin_runs/n2cs/py.sh -m hydrophysics.twin.forward --theta {K.R1C_THETA} "
            f"--vep-json {K.R1C_COLUMN} {flags}")
    a = "".join([f" --recorded-pumping 'UCH={K.UNIT_CH}@2023-01'",
                 f" --recorded-pumping 'UYL={K.UNIT_YL}@2023-01'"]
                + [f" --recorded-pumping 'RCH{y}={K.REPLAYS}/rch{y}.npz@2023-01'"
                   for y in K.REPLAY_YEARS])
    b = "".join(f" --recorded-pumping 'RYL{y}={K.REPLAYS}/ryl{y}.npz@2023-01'"
                for y in K.REPLAY_YEARS)
    return {"A": base + a + f" --out {K.FWD_A}", "B": base + b + f" --out {K.FWD_B}"}


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def draft_osse_declaration(seed: int, *, params: dict | None = None) -> dict:
    """N2CS.osse_design.v1 (harness.prereg3, kind 'declaration'), to be registered by the lead
    BEFORE ``run_n2cs.sh files``. Inputs list only files that exist now."""
    design = draft_osse_record(seed, params=params)
    code = code_sha256()
    inputs = ([{"path": p, "sha256": h} for p, h in code.items() if h != "missing"]
              + [p for p in DECL_INPUTS if rp(p).exists()])
    return {
        "id": DECL_ID, "wp": K.WP, "kind": "declaration",
        "title": "N2CS OSSE design: synthetic decidability gate (G0) and estimator-form choice "
                 "for the prospective Changhua first-year size-class test; model-only",
        "priors": "N2CS (round-8 #6, user ruling R8.2) will regress e = obs - ZR at N2C's 74 "
                  "Changhua wells on two R1c contrasts (Changhua-only and Yunlin-only recorded "
                  "anomalies). Before any replay forward runs, this declaration freezes the "
                  "synthetic worlds, the three estimator forms, the G0 gate, the choice rule and "
                  "the seeds, so the form is chosen by the OSSE only.",
        "rulings": {K.RULING_ID: {"file": K.RULINGS, "sha256": K.RULINGS_SHA}},
        "design": design, "code_sha256": code,
        "recipe": {"code": f"main {K.WT_HEAD} (N2C's code commit) in the clean detached "
                           f"worktree {K.WT}", "forwards": forward_commands(),
                   "cpu": "two CPU forwards (10 and 8 scenarios, hours), then minutes of "
                          "single-threaded replicates",
                   "runner": f"bash {K.X}/run_n2cs.sh files, forward-a, forward-b, extract, "
                             "osse"},
        "zones": "N2C's 74 Changhua wells (screens of dataprep_staging/x5_class_share/"
                 f"wells_frozen.json; 5 km blocks from the x, y of {K.WELLS_XY})",
        "burn_in": _na("synthetic; the AR(1) noise starts stationary"),
        "folds": _na("synthetic declaration; no folds"),
        "gate_scores": _na("declared G0 thresholds; no weighted score"),
        "primary_model": "R1c (stage3_datum_gate_ccz, main set) supplies the contrasts; the "
                         "characterised object is N2CS's estimator and readings",
        "inputs": inputs,
        "tests": _na("synthetic declaration: no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "thresholds": {"g0": K.G0, "choice_rule": K.CHOICE_RULE,
                       "max_filled_fraction": K.MAX_FILLED_FRACTION,
                       "pregate_tol_m": K.PREGATE_TOL_M, "plumbing_tol": K.PLUMBING_TOL},
        "attempt_policy": {
            "scored_runs": "one extract and one OSSE run; result.json written once (mode 'x') "
                           "and final, whatever it shows",
            "sealing": SEALING,
            "engineering_retries": "free before result.json exists: a crash, a missing file, a "
                                   "failed pre-gate or a failed EXACT plumbing check; each "
                                   "logged in attempts.jsonl; a fix that changes bound code "
                                   "needs a v2 that discloses the failed attempts",
            "v2_scope": "code only; never PARAMS, K_GRID, GATE_VARIANTS, REPLICATES, the seed, "
                        "G0, the choice rule, the forms or the replays",
            "panel": "extract appends the panel sha256 to attempts.jsonl; run refuses any other "
                     "panel", "exploratory_followups_max": 1, "thresholds_frozen": True},
        "interpretation": "G0 says whether N2CS can decide 'overstated >= x2' vs 'not "
                          "overstated by x2' by design; if no form passes, N2CS is not "
                          "registered and the N line is bookkeeping. Nothing here is evidence "
                          "about the aquifer.",
        "rule_risks_disclosure": [
            DESIGN_DEVICE,
            "the census source is pump_electricity/preorigin/v3 (not model_ready/"
            "electricity_census/v1, which X5.n2c_reading used)",
            "v3's exemption: its 2026 census attribute table includes pumps installed after "
            "the origin (Yunlin 44, Changhua 34; in-grid 2012-2022 energy change <= 3.7e-4 "
            "relative, MANIFEST C0.v3.exemption.census_attributes.preorigin), allowed for "
            "design_time_evidence; this is a post-origin attribute input against U7, disclosed",
            "the county-uniform confounder is absorbed exactly by the month effects (by "
            "construction, and so is any real county-uniform term); the shaped variants carry "
            "the confounder test",
            "G0 uses k = 1 and k = 1/3 only; the other k are report-only",
            "power may depend on the true Yunlin coefficient (report cells k_YL = 3); both "
            "intervals (pinned and refit) hold the month effects fixed across draws",
            "winner's curse: the chosen form maximises power over three forms evaluated on "
            "common worlds; the chosen power is optimistic by about the Monte Carlo SE "
            "(about 0.009 at 0.8 with 2000 replicates)",
            "the builder ran smoke runs of the engine on a synthetic stand-in panel (no model "
            "output); no parameter was changed after them",
            "the noise parameters are N2C-reading's published design-time values "
            "(synth_n2c.PARAMS) from spent pre-origin heads"],
        "no_prior_look_statement": "No N2CS replay or forward exists; no replicate on R1c's "
                                   "two contrasts has been drawn; no 2026-12+ head exists.",
        "written_by": f"{K.X}/osse_n2cs.py"}


def validate_declaration(doc: dict) -> list[str]:
    from harness import prereg3

    return prereg3.validate_prereg(doc)


# ----------------------------------------------------------------------------- run
def last_extract_panel_sha(attempts: Path | None = None) -> str | None:
    p = attempts or rp(K.ATTEMPTS)
    if not p.exists():
        return None
    ev = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    ex = [r for r in ev if r.get("event") == "extract" and r.get("pregate_ok")]
    return ex[-1]["panel_sha256"] if ex else None


def run() -> dict:
    bad = declared_code_problems()
    if bad:
        raise SystemExit("REFUSED: " + "; ".join(bad))
    from harness import prereg3

    decl, decl_sha = prereg3.load_prereg(DECL_ID)
    want = last_extract_panel_sha()
    if want is None or sha256(rp(K.PANEL)) != want:
        raise SystemExit("REFUSED: the panel is not the one extract logged in attempts.jsonl")
    out = rp(K.OSSE_OUT) / "result.json"
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists; the OSSE runs once")
    design = decl["design"]
    log_attempt({"event": "osse_start", "declaration_sha256": decl_sha, "panel_sha256": want})
    res = L.run_osse(load_panel(rp(K.PANEL)), design["params"],
                     seed=int(design["seeds"]["world_seed"]))
    res.update({"declaration": DECL_ID, "declaration_sha256": decl_sha, "panel_sha256": want,
                "written": utc()})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1, default=float) + "\n")
    s = sha256(out)
    log_attempt({"event": "osse_run", "result_sha256": s,
                 "engineering_failure": res["engineering_failure"],
                 "passed": (res.get("choice") or {}).get("passed")})
    return {"result": str(out), "sha256": s, "choice": res.get("choice")}


def smoke(seed: int, out: Path, reps: int = 20, B: int = 200) -> dict:
    """Synthetic stand-in panel, few replicates, small B: plumbing of the engine only."""
    panel = L.synthetic_panel(seed)
    params = {"sigma_tot_m": 0.3, "block_share": 0.44, "phi": 0.86, "kappa_m": 0.1}
    res = L.run_osse(panel, params, seed=seed, reps=reps, B=B)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_result.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return {"smoke": True, "choice": res.get("choice")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    r = sub.add_parser("declare")
    r.add_argument("--seed", type=int, required=True)
    sub.add_parser("run")
    s = sub.add_parser("smoke")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        r = extract()
        print(json.dumps({"pregate_ok": r["pregate"]["ok"], "panel_sha256": r["panel_sha256"]}))
    elif a.cmd == "declare":
        d = draft_osse_declaration(a.seed)
        missing = [p for p in DECL_INPUTS if not rp(p).exists()]
        print(json.dumps({"id": d["id"], "errors": validate_declaration(d),
                          "inputs_missing_dropped": missing,
                          "registry_state": declared_code_problems() or "registered, code "
                                                                        "unchanged"},
                         indent=1))
    elif a.cmd == "run":
        print(json.dumps(run(), default=float))
    else:
        print(json.dumps(smoke(a.seed, a.out), default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
