"""Run test G1 of ``N1.phase12_response.v1`` (the masked 2015 refit) and report G3.

G1: the R0 recipe (batch-8 cwa) refitted with ``calibrate_flow --mask-months 2015-01:2015-05``
(batch 23, ``results/twin_runs/stage3_datum_gate_m15``). The I1 primary model is R0 (no I1 rung
passed: results/dataprep/reports/I1_Rstar), so the two registered models are one refit. Its
rollout M on the CPU engine, and ZR = the same rollout with the Jan-May 2015 pumping energy
replaced cell by cell by the reference-year mean of the same calendar month (R0 has no canal
delivery term; rain unchanged). Scored at the 158 calibration wells with ``run_cpu_tests``'s
own anomaly, eligibility and paired site-bootstrap code (seed 20261024).

G3 needs a recipe with a canal-delivery term; R0 has none, so G3 is reported 'not run'.

1. ``prereg3.assert_can_score`` for G1 BEFORE any head is read (logs the scoring look).
2. The refit theta is checked: mask_months recorded, every other recipe key equal to cwa's.
3. Observed heads from ``heads/L1_clean/v3`` (guard, scoring_target), the n1lib scores.

``--dry`` runs steps 2-3 on synthetic heads (no look, nothing written): a code check; before
the refit exists it uses the cwa theta.
Outputs: ``results/dataprep/reports/N1_evidence/phase12_tests/`` (results.json, g1_wells.csv).

Usage: CUDA_VISIBLE_DEVICES="" cpu_slot.sh python -I run_phase12_tests.py [--dry]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import run_cpu_tests as C  # noqa: E402
from n1_natural_experiments import units_n1 as U  # noqa: E402

PREREG = "N1.phase12_response.v1"
G1, G3 = "N1.G1_mask_E2015", "N1.G3_sw_E2015"
SEED_G1 = 20261024                       # the test's registered seed
MASK = "2015-01:2015-05"
REFIT = REPO / "results/twin_runs/stage3_datum_gate_m15/stage3_theta.json"
OUT = K.DATAPREP / "reports/N1_evidence/phase12_tests"
# meta keys that may differ between the refit and the cwa fit (outputs, not recipe)
NOT_RECIPE = {"git_commit", "r2_insample", "bounds_hit", "well_datum", "well_datum_stats",
              "mask_months", "gate"}


def load_refit(theta: Path = REFIT, check: bool = True):
    import torch
    from p4_changhua import engine, inputs_p4

    from hydrophysics.twin.forward import build_model, load_members

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing")
    os.chdir(REPO)
    member = load_members([str(theta)])[0]
    ref = load_members([str(K.TWIN_THETA)])[0]
    if check and member.meta.get("mask_months") != MASK:
        raise SystemExit(f"{REFIT}: mask_months {member.meta.get('mask_months')!r}, "
                         f"the test needs {MASK!r}")
    diff = sorted(k for k in set(member.meta) | set(ref.meta) if k not in NOT_RECIPE
                  and member.meta.get(k) != ref.meta.get(k))
    if diff:
        raise SystemExit(f"the refit's recipe differs from cwa's in {diff}")
    inp = inputs_p4.load_inputs(member.meta)
    E = np.stack([inp.E_by_class[c] for c in inp.E_by_class]).sum(axis=0)[:, 1:]
    R = inp.recharge_field.numpy()[:, 1:]
    ge = inp.ground_elev.numpy()
    h0 = inp.initial_heads(0, n_layers=4).numpy()
    model, sc, _ = build_model(inp.grid, member, "cpu")
    tw = engine.SparseTwin(model, sc, ge, pump_layer=int(member.meta.get("pump_layer", 1)),
                           recharge_layer=int(member.meta.get("recharge_layer", 0)))
    return {"tw": tw, "E": E, "R": R, "h0": h0, "inp": inp, "member": member}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="code check on synthetic heads")
    a = ap.parse_args(argv)
    if not a.dry:
        OUT.mkdir(parents=True, exist_ok=True)
        if (OUT / "results.json").exists():
            raise SystemExit(f"{OUT / 'results.json'} exists; results are never recomputed")
        r = prereg3.assert_can_score(PREREG, K.WP, test_id=G1)
        scored = {"sha256": r["sha256"], "inputs_verified": r["inputs_verified"],
                  "post_registration_looks": len(r["post_registration_looks"]),
                  "unverifiable_prior_looks": len(r["unverifiable_prior_looks"])}
        C.log(f"assert_can_score {G1}: ok ({scored})")

    # --dry before the refit exists: the cwa theta exercises the same code path
    mdl = load_refit(REFIT, True) if not a.dry or REFIT.exists() else load_refit(K.TWIN_THETA,
                                                                                 False)
    tw, E, R, h0 = mdl["tw"], mdl["E"], mdl["R"], mdl["h0"]
    H = {"M": tw.rollout(h0, E=E, R=R), "ZR": tw.rollout(h0, E=C.zr_energy(E, "E2015"), R=R)}
    C.log(f"rollouts done: {sorted(H)}")

    cal = U.calibration_wells()
    meta = U.station_meta().set_index("sid")
    cal["cell"] = U.cell_index(cal.x.to_numpy(), cal.y.to_numpy())
    cal["layer"] = cal.layer.astype(int) - 1
    cal["site"] = cal.sid.str[:6]
    cal["county"] = cal.sid.map(meta.CountyName).map({"彰化縣": "Changhua", "雲林縣": "Yunlin",
                                                       "南投縣": "Nantou"})
    inp = mdl["inp"]
    oi = {s: (int(c), int(lay)) for s, c, lay in zip(inp.sids, inp.obs_idx, inp.obs_layer,
                                                      strict=True)}
    conv = all(oi[s] == (c, lay) for s, c, lay in zip(cal.sid, cal.cell, cal.layer, strict=True))
    if not conv:
        raise SystemExit("calibration-well cell/layer convention differs from obs_idx/obs_layer")
    sim = {k: C.heads_at(H[k], cal.cell.to_numpy(), cal.layer.to_numpy()) for k in ("M", "ZR")}

    if a.dry:
        rng = np.random.default_rng(0)
        obs = sim["M"] + rng.normal(0.0, 0.5, sim["M"].shape)
        obs[:, :12] = np.nan                                    # 2012 is never scored
    else:
        obs = C.observed(list(cal.sid))
    tab = C.event_table(obs, sim, cal, "E2015")
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "prereg": PREREG, "refit_theta": rel_to_repo(REFIT), "mask_months": MASK,
           "models_note": "R0 recipe = the I1 primary model (no I1 rung passed); one refit",
           "tests": {
               G1: {"M_vs_CL": C.compare(tab, "M", "CL", SEED_G1),
                    "M_vs_ZR": C.compare(tab, "M", "ZR", SEED_G1),
                    "zr_informative_rms_m": C.informative(tab, "ZR"),
                    "slope_M": C.slope(tab, "M", SEED_G1),
                    "by_county": {c: {"M_vs_CL": C.compare(g, "M", "CL", SEED_G1)}
                                  for c, g in tab.groupby("county")}},
               G3: {"verdict": "not run",
                    "reason": "no recipe of G1 carries a canal-delivery term (R0 has none)"}}}
    g1 = res["tests"][G1]
    if g1["zr_informative_rms_m"] < K.ZR_INFORMATIVE_M:
        g1["M_vs_ZR"]["verdict_registered"] = "not informative"
    C.log(f"G1: M vs CL {g1['M_vs_CL'].get('verdict')} ({g1['M_vs_CL'].get('rmse_model')}, "
          f"{g1['M_vs_CL'].get('rmse_base')}); M vs ZR {g1['M_vs_ZR'].get('verdict')}; "
          f"ZR informative rms {g1['zr_informative_rms_m']:.3f} m; {len(tab)} wells")
    if a.dry:
        print(json.dumps(res, indent=1, default=str)[:3000])
        return 0
    res["prereg_scored"] = scored
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    tab.to_csv(OUT / "g1_wells.csv", index=False)
    (OUT / "run.log").write_text("\n".join(C.LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
