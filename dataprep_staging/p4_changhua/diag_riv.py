"""P4.0 diagnostic: where does the sparse-vs-torch difference of the RIV configuration come
from? (Run after build_engine; evidence only, no product.)

    cd dataprep_staging
    CUDA_VISIBLE_DEVICES= tools/cpu_slot.sh --slots 2 python -m p4_changhua.diag_riv OUT.json

1. connection masks of both rollouts, step by step (a different RIV switch decision?);
2. torch's per-step solve error: from torch's own previous head, the exact sparse one-step
   solution vs torch's next head (the CG tolerance of the reference, relative 1e-8 of the
   right-hand side);
3. the same comparison with the reference's CG tolerance tightened in this process only
   (``flow._cg.__defaults__``; no file is edited), for the RIV and the base configuration.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from common.paths import REPO  # noqa: E402

from p4_changhua import build_engine as be  # noqa: E402
from p4_changhua import engine, inputs_p4  # noqa: E402

torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))


def run_case(tw, sc, h0, E, R, sw, label):
    from hydrophysics.twin import flow

    out = {}
    H_sp = tw.rollout(h0, E=E, R=R, sw=sw, month0=1)
    for tol, maxit in ((1e-8, 2000), (1e-12, 20000)):
        defaults = flow._cg.__defaults__
        flow._cg.__defaults__ = (None, None, tol, maxit)
        try:
            t0 = time.perf_counter()
            r = engine.verify_against_torch(tw, sc, h0, E, R, sw=sw, month0=1, H_sparse=H_sp)
            r["t_torch_s"] = round(time.perf_counter() - t0, 1)
        finally:
            flow._cg.__defaults__ = defaults
        H_t = r.pop("H_torch")
        rec = {k: v for k, v in r.items()}
        if tw.has_rivers:
            ms = H_sp[tw.riv_layer][tw.riv_idx][:, :-1] > tw.riv_rbot[:, None]
            mt = H_t[tw.riv_layer][tw.riv_idx][:, :-1] > tw.riv_rbot[:, None]
            dis = ms != mt
            rec["mask_disagreements"] = int(dis.sum())
            if dis.any():
                gap = np.abs(H_sp[tw.riv_layer][tw.riv_idx][:, :-1] - tw.riv_rbot[:, None])
                rec["max_head_minus_rbot_at_disagreement_m"] = float(gap[dis].max())
        # torch's per-step solve error from its own previous state
        err = []
        for t in range(E.shape[1]):
            one = tw.rollout(H_t[..., t], E=E[:, t:t + 1], R=R[:, t:t + 1],
                             sw=None if sw is None else sw[:, t:t + 1], month0=1 + t,
                             apex_from=h0, n_steps=1)
            err.append(float(np.abs(one[..., 1] - H_t[..., t + 1]).max()))
        rec["torch_one_step_error_max_m"] = float(max(err))
        rec["torch_one_step_error_median_m"] = float(np.median(err))
        out[f"cg_tol_{tol:g}"] = rec
        print(label, tol, json.dumps(rec), flush=True)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_path = Path(argv[0]).resolve()
    os.chdir(REPO)
    from hydrophysics.twin.forward import load_members

    member = load_members([str(be.THETA)])[0]
    inp = inputs_p4.load_inputs(member.meta)
    grid = inp.grid
    ge = inp.ground_elev.numpy()
    h0 = inp.initial_heads(0, n_layers=4).numpy()
    E = np.stack([inp.E_by_class[c] for c in inp.E_by_class]).sum(axis=0)[:, 1:]
    R = inp.recharge_field.numpy()[:, 1:]
    nT, A = E.shape[1], grid.n_active
    res = {"theta_sha256": be.sha256_file(be.THETA)}
    _, sc, _, tw = be.build(member, grid, ge)
    res["base"] = run_case(tw, sc, h0, E, R, None, "base")
    mr = be.riv_member(member)
    _, sc_r, zone3_r, tw_r = be.build(mr, grid, ge)
    months = np.arange(1, nT + 1) % 12
    sw = np.zeros((A, nT))
    sw[zone3_r == 1] = 2e-3 * np.clip(np.sin(2 * np.pi * (months - 2) / 12.0), 0, None)[None, :]
    res["riv"] = run_case(tw_r, sc_r, h0, E, R, sw, "riv")
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
