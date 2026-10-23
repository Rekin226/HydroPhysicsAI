"""WP S3 acceptance run (pre-registration ``S3.gnss_clean_acceptance.v1``).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/run_validation.py \
        --variant {preorigin,full} --tune TUNE_JSON --out EVIDENCE_DIR

``preorigin`` (the gate, U7): every series cut at 2022-12-31; A1 injected steps (seed 303,
2/3/5/10 cm x 10 repetitions, WRA and NGL), A2 the 2021 extra compaction at co-located
rings and piles after the common-mode removal, common-mode scatter and PCA checks.

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
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_tables as BT  # noqa: E402
import injection as INJ  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
import validate as V  # noqa: E402
from harness import prereg  # noqa: E402

PREREG_ID = "S3.gnss_clean_acceptance.v1"
A1_TEST = "S3.A1_injected_steps_preorigin"
SEED = 303
AMPS = (0.02, 0.03, 0.05, 0.10)
N_REP = 10


def load_tuned(path: str) -> tuple[M.Params, list[str], dict]:
    t = json.loads(Path(path).read_text())
    cp = t["chosen_params"]
    p = replace(M.Params(), **{k: cp[k] for k in M.Params().as_dict()})
    return p, list(t["reference"]), t


A2_UNITS = ["s3.pair.GFES-光復國小", "s3.pair.HLES-宏崙國小", "s3.pair.HNES-湖南國小",
            "s3.pair.JMES-拯民國小", "s3.pair.KTES-客厝國小", "s3.pair.SJES-溪州國小",
            "s3.pair.SSES-新生國小", "s3.pair.STES-秀潭國小", "s3.pair.TKJS-土庫國中",
            "s3.pair.YCES-元長國小", "s3.pair.GFES-NTUH_282_29", "s3.pair.SJES-SJES",
            "s3.pair.STES-STES", "s3.pair.TKJS-TKJS_281_29"]


def log_looks(variant: str, skip_injection: bool) -> list[dict]:
    """Pre-registration check and look log, BEFORE any score is computed. The pre-origin A1
    (the registered test) goes through ``prereg.assert_can_score``, which refuses to score if
    the pre-registration postdates a look at its units and logs the scoring look; every other
    score computed here (the declared A2, the reported analyses) is logged with
    ``prereg.record_look`` under the same pre-registration id."""
    out = []
    prereg.load_prereg(PREREG_ID)            # exists and unchanged, or refuse to run
    if variant == "preorigin":
        if not skip_injection:
            r = prereg.assert_can_score(PREREG_ID, "S3", test_id=A1_TEST)
            out.append({"test_id": A1_TEST, "assert_can_score": "ok",
                        "other_tests_prior_looks": r["other_tests_prior_looks"]})
            lk = prereg.record_look(["s3.injection.harnessG.r0-9.wra.preorigin",
                                     "s3.injection.harnessG.r0-9.ngl.preorigin"],
                                    "gnss_injection:preorigin_background:harness_family_G",
                                    "S3", "score", prereg_id=PREREG_ID,
                                    note="A1 secondary (reported, not gating)")
            out.append({"time": lk["time"], "data_scope": lk["data_scope"]})
        lk = prereg.record_look(A2_UNITS, "gnss_wra+rings+piles:2020-01..2021-12:extra_2021",
                                "S3", "score", prereg_id=PREREG_ID,
                                note="A2 declared criterion, pre-origin variant (gate)")
        out.append({"time": lk["time"], "data_scope": lk["data_scope"]})
    else:
        units = A2_UNITS + ["s3.twin_score.gnss_rates.wra.2023-01..2025-03",
                            "s3.twin_score.gnss_rates.ngl.2023-01..2024-04"]
        if not skip_injection:
            units += ["s3.injection.seed303.wra.full", "s3.injection.seed303.ngl.full",
                      "s3.injection.harnessG.r0-9.wra.full", "s3.injection.harnessG.r0-9.ngl.full"]
        lk = prereg.record_look(units, "gnss_full_records:reported_not_gating", "S3", "score",
                                prereg_id=PREREG_ID,
                                note="full variant: A1 full records, A2 full, A3, twin scoring "
                                     "(after-the-fact, never a gate; U7)")
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
    p, ref, _ = load_tuned(a.tune)
    end = P.ORIGIN_END if a.variant == "preorigin" else None
    looks = log_looks(a.variant, a.skip_injection)
    tab = BT.build(p, ref, end, version=f"gnss_clean/{a.variant}/v1")
    print(f"tables built ({time.time() - t0:.0f}s)", flush=True)
    (od / f"tables_{a.variant}.json").write_text(json.dumps(BT.table_hashes(tab), indent=1))
    res: dict = {"variant": a.variant, "params": p.as_dict(), "reference": ref,
                 "window_end": None if end is None else str(end.date()),
                 "prereg": PREREG_ID, "looks_logged": looks}
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
        # secondary (reported): the A1 harness generator, family G, r = 0..9
        g = []
        wser = P.wra_series(end)
        bw = P.run_wra(p, window_end=end, series=wser)
        g.append(INJ.run_trials_harness_g("wra", p, bw, end, range(10), series=wser))
        nser, _ = P.ngl_series(end)
        bn = P.run_ngl(p, ref=ref, window_end=end, series=nser)
        g.append(INJ.run_trials_harness_g("ngl", p, bn, end, range(10), ref=ref,
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
