"""WP S3 acceptance run, v2 code (pre-registration ``S3.gnss_clean_acceptance.v2``, repair
round 2026-10-12/13; looks and the scoring check use ``harness.prereg3``, the tool A1 repair
round 2 prescribes, with members declared for every S3 unit).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v2/run_validation.py \
        --variant {preorigin,full} --tune TUNE_JSON --out EVIDENCE_DIR

``preorigin`` (the gate, U7): every series cut at 2022-12-31, pre-origin station sets
(``pipeline.wra_policy``, ``pipeline.ngl_policy``); A1 injected steps (seed 404, new units;
2/3/5/10 cm x 10 repetitions, WRA and NGL), A2 the 2021 extra compaction at co-located rings
and piles after the removal of each offered common mode (``ngl_offfan`` and ``wra_hf``),
common-mode scatter and PCA checks; secondary harness family G with r = 10..19 (r = 0..9
were used by v1).

``full`` (reported, after-the-fact; never a gate): the same on the full records (WRA
2020-01..2025-03, NGL to 2026-09), A3 excess-rate stability, and the scoring of the frozen
forward archive against GNSS rates 2023-01..2025-03 (WRA) and 2023-01..2024-04 (NGL), raw
and common-mode removed.

Writes ``<out>/validation_<variant>.json``, the injection trial table and
``tables_<variant>.json`` (pandas hashes of the product tables, checked by ``build.py``).
Hyperparameters and the reference set come from the pre-origin tuning file (``--tune``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # dataprep_staging
sys.path.insert(0, str(Path(__file__).resolve().parent))       # v2 modules first
import build_tables as BT  # noqa: E402
import injection as INJ  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
import units_v2 as LK  # noqa: E402
import validate as V  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "S3.gnss_clean_acceptance.v2"
A1_TEST = "S3.A1_injected_steps_preorigin_v2"
SEED = 404
AMPS = (0.02, 0.03, 0.05, 0.10)
N_REP = 10
HARNESS_G_R = range(10, 20)
VERSION = "v2"


def load_tuned(path: str, variant: str) -> tuple[M.Params, list[str], dict]:
    """Tuned detector and the variant's off-fan reference set (pre-origin: ``reference``;
    full: ``reference_full``, the same rule over the full candidate list)."""
    t = json.loads(Path(path).read_text())
    cp = t["chosen_params"]
    p = replace(M.Params(), **{k: cp[k] for k in M.Params().as_dict()})
    ref = t["reference"] if variant == "preorigin" else t["reference_full"]
    return p, list(ref), t


def cpu_env() -> dict:
    """Evidence that the run went through tools/cpu_slot.sh (single-threaded BLAS, nice 10)."""
    return {**{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "nice": os.nice(0)}


A2_UNITS = LK.A2_UNITS
TWIN_UNITS = {
    "s3.twin_score.gnss_rates.wra.2023-01..2025-03": {
        "ids": ["gnss:wra_network_on_grid"], "window": "2023-01..2025-03",
        "description": "subsidence rates of the on-grid WRA GNSS stations against the frozen "
                       "forward archive (after-the-fact)"},
    "s3.twin_score.gnss_rates.ngl.2023-01..2024-04": {
        "ids": ["gnss:ngl_on_grid"], "window": "2023-01..2024-04",
        "description": "subsidence rates of the on-grid NGL stations against the frozen "
                       "forward archive (after-the-fact)"}}


def log_looks(variant: str, skip_injection: bool) -> list[dict]:
    """Pre-registration check and look log, BEFORE any score is computed. The pre-origin A1
    (the registered test) goes through ``prereg3.assert_can_score``, which refuses to score if
    the pre-registration postdates a look at its units (or its registered inputs changed) and
    logs the scoring look; every other score computed here (the declared A2, the reported
    analyses) is logged with ``prereg3.record_look`` under the same pre-registration id."""
    out = []
    prereg3.load_prereg(PREREG_ID)           # exists and unchanged, or refuse to run
    if variant == "preorigin":
        if not skip_injection:
            r = prereg3.assert_can_score(PREREG_ID, "S3", test_id=A1_TEST)
            out.append({"test_id": A1_TEST, "assert_can_score": "ok",
                        "inputs_verified": r.get("inputs_verified"),
                        "post_registration_looks": len(r.get("post_registration_looks", [])),
                        "unverifiable_prior_looks": len(r.get("unverifiable_prior_looks",
                                                              []))})
            gu = ["s3.injection.harnessG.r10-19.wra.preorigin",
                  "s3.injection.harnessG.r10-19.ngl.preorigin"]
            lk = prereg3.record_look(gu, "gnss_injection:preorigin_background:harness_family_G"
                                         "_r10-19", "S3", "reported_score",
                                     members=LK.seed_members(gu), prereg_id=PREREG_ID,
                                     note="A1 secondary, v2 (reported, not gating)")
            out.append({"time": lk["time"], "data_scope": lk["data_scope"]})
        lk = prereg3.record_look(A2_UNITS, "gnss_wra+rings+piles:2020-01..2021-12",
                                 "S3", "declared_criterion_score",
                                 members=LK.a2_members("2020-01..2021-12"), prereg_id=PREREG_ID,
                                 note="A2 declared criterion (v2: each offered common mode), "
                                      "pre-origin variant (gate)")
        out.append({"time": lk["time"], "data_scope": lk["data_scope"]})
    else:
        units = A2_UNITS + list(TWIN_UNITS)
        members = {**LK.a2_members("2020-01..2025-03"), **TWIN_UNITS}
        if not skip_injection:
            inj = ["s3.injection.seed404.wra.full", "s3.injection.seed404.ngl.full",
                   "s3.injection.harnessG.r10-19.wra.full",
                   "s3.injection.harnessG.r10-19.ngl.full"]
            units += inj
            members.update(LK.seed_members(inj))
        lk = prereg3.record_look(units, "gnss_full_records:2007-01..2026-09", "S3",
                                 "after_the_fact", members=members, prereg_id=PREREG_ID,
                                 note="full variant v2: A1 full records, A2 full, A3, twin "
                                      "scoring (after-the-fact, never a gate; U7)")
        out.append({"time": lk["time"], "data_scope": lk["data_scope"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["preorigin", "full"], required=True)
    ap.add_argument("--tune", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-injection", action="store_true", help="smoke test only")
    a = ap.parse_args(argv)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p, ref, _ = load_tuned(a.tune, a.variant)
    end = P.ORIGIN_END if a.variant == "preorigin" else None
    looks = log_looks(a.variant, a.skip_injection)
    tab = BT.build(p, ref, end, version=f"gnss_clean/{a.variant}/{VERSION}")
    print(f"tables built ({time.time() - t0:.0f}s)", flush=True)
    (od / f"tables_{a.variant}.json").write_text(json.dumps(BT.table_hashes(tab), indent=1))
    res: dict = {"variant": a.variant, "params": p.as_dict(), "reference": ref,
                 "window_end": None if end is None else str(end.date()),
                 "prereg": PREREG_ID, "looks_logged": looks, "cpu_env": cpu_env(),
                 "wra_stations": sorted(tab["wra_daily"].station.unique()),
                 "ngl_stations": sorted(tab["ngl_daily"].station.unique())}
    wd = tab["wra_daily"]
    cms = tab["cms"]
    st = tab["stations"]
    members = tab["wra_members"]
    # A2: the 2021 extra compaction survives the common-mode removal
    res["A2_2021_extra"] = V.test_2021_extra(wd, cms, st)
    print("A2", json.dumps(BT.clean_json({k: v for k, v in res["A2_2021_extra"].items()
                                          if k != "pairs"}), default=str), flush=True)
    # common-mode diagnostics
    t_end = "2022-12-31" if end is not None else "2025-03-31"
    res["cm_scatter_wra"] = V.scatter_table(wd, "resid", cms, members,
                                            ["ngl_offfan", "wra_hf", "wra_onfan_fullband"],
                                            "2020-01-01", t_end)
    nd = tab["ngl_daily"]
    ngl_fan = [k for k in P.NGL_FAN + P.NGL_GRID_EXTRA if k not in ref]
    res["cm_scatter_ngl_fan"] = V.scatter_table(nd, "resid", cms, ngl_fan, ["ngl_offfan"],
                                                "2007-01-01", "2026-12-31")
    res["cm_scatter_ngl_ref_loo_note"] = ("reference stations are in their own median; "
                                          "their scatter is not an independent check")
    res["pca_wra"] = V.pca_check(wd, members, cms, p.hp_window)
    res["steps_summary"] = BT.steps_summary(tab["steps"])
    if a.variant == "full":
        res["A3_excess_stability"] = V.excess_stability(wd, cms, st)
        res["A2_note"] = "full-variant A2 is reported only; the gate uses the pre-origin A2"
        res["twin_scores_wra"] = V.twin_scores(wd, cms, st, "wra", "2023-01-01", "2025-03-31",
                                               members=members)
        res["twin_scores_ngl"] = V.twin_scores(nd, cms, st, "ngl", "2023-01-01", "2024-04-30")
    if not a.skip_injection:
        inj, trials = V.injection_acceptance(p, ref, end, SEED, AMPS, N_REP, 0.90)
        trials.to_csv(od / f"injection_trials_{a.variant}.csv", index=False)
        res["A1_injected_steps"] = inj
        print("A1", json.dumps(BT.clean_json(inj), default=str), flush=True)
        # secondary (reported): the A1 harness generator, family G, r = 10..19
        g = []
        wser = P.wra_series(end)
        bw = P.run_wra(p, window_end=end, series=wser)
        g.append(INJ.run_trials_harness_g("wra", p, bw, end, HARNESS_G_R, series=wser))
        nser, _ = P.ngl_series(end)
        bn = P.run_ngl(p, ref=ref, window_end=end, series=nser)
        g.append(INJ.run_trials_harness_g("ngl", p, bn, end, HARNESS_G_R, ref=ref,
                                          series=nser))
        G = pd.concat(g)
        G.to_csv(od / f"injection_harness_g_{a.variant}.csv", index=False)
        res["A1_secondary_harness_family_G"] = {
            fam: {"n": int(len(t)), "recovered": float(t.recovered.mean()),
                  "found_within_3d": float(t.found.mean()),
                  "recovered_dense_enough": float(t[t.dense_enough].recovered.mean())
                  if t.dense_enough.any() else None,
                  "n_dense_enough": int(t.dense_enough.sum())}
            for fam, t in G.groupby("family")}
        print("A1-G", json.dumps(BT.clean_json(res["A1_secondary_harness_family_G"])),
              flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    (od / f"validation_{a.variant}.json").write_text(
        json.dumps(BT.clean_json(res), indent=1, allow_nan=False, default=str))
    print(f"done ({res['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
