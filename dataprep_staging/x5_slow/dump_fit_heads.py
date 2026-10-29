"""Save the nudged hindcast heads calibrate_coupled fits the column on (GPU, about 1-2 min),
so the synthetic truths and the fits share one head field.

It does not re-implement the hindcast: it runs ``calibrate_coupled.main`` itself with the
theta and the flags of every E2 arm (``SPEC["calibrate_common"]``; the form and withhold
flags do not enter the hindcast), captures what ``hindcast_with_nudging`` returns, and stops
main right there by an exception, before main reads any leveling or creates its --out dir.
The heads are written in their native dtype, months <= SPEC["heads"]["last_month"] only,
with the forward npz's keys (heads_mean (1, L, A, T), dates, scenario_names, mask, x0, y0,
dx) plus provenance_json (theta and code hashes, the argv, the head inputs). It prints no
head value.

Run it with the frozen code (results/twin_runs/par32/py.sh), from the repository root:

    HYDROMIND_GW_DATA=$PWD/chou-shui-data/data results/twin_runs/par32/py.sh \\
        dataprep_staging/x5_slow/dump_fit_heads.py

It refuses to overwrite an existing npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as C  # noqa: E402


class _Captured(Exception):
    """Raised inside calibrate_coupled.main right after the hindcast."""


def hindcast_argv(data_dir: str, out_dir: str) -> list[str]:
    """calibrate_coupled's argv up to the hindcast: the theta and the arms' common flags."""
    return (["--theta", C.SPEC["theta"]] + list(C.SPEC["calibrate_common"])
            + ["--data", str(data_dir), "--out", str(out_dir)])


def capture(cc, argv: list[str]) -> tuple[np.ndarray, object, dict]:
    """Run ``cc.main(argv)`` until its ``hindcast_with_nudging`` returns; give back that
    result as numpy (L, A, T), the TwinInputs it was called with, and the call's keyword
    arguments (for the provenance)."""
    real = cc.hindcast_with_nudging
    box: dict = {}

    def spy(model, scalars, inp, *args, **kw):
        out = real(model, scalars, inp, *args, **kw)
        box.update(heads=out.detach().cpu().numpy(), inp=inp,
                   call={"gain": args[3] if len(args) > 3 else kw.get("gain"),
                         "every": args[4] if len(args) > 4 else kw.get("every"),
                         "update_method": kw.get("update_method"),
                         "apex_from_given": kw.get("apex_from") is not None,
                         "datum_given": kw.get("datum") is not None})
        raise _Captured

    cc.hindcast_with_nudging = spy
    try:
        cc.main(argv)
    except _Captured:
        pass
    finally:
        cc.hindcast_with_nudging = real
    if "heads" not in box:
        raise SystemExit("calibrate_coupled.main finished without a nudged hindcast: the "
                         "flags do not select hindcast_with_nudging")
    return box["heads"], box["inp"], box["call"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(C.repo_path(C.HEADS_NPZ)))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; the truth heads are written once")
    from hydrophysics.twin import calibrate_coupled as cc

    theta = C.repo_path(C.SPEC["theta"])
    meta = json.loads(theta.read_text())["meta"]
    if meta.get("wells_dir") != C.SPEC["heads"]["wells_dir"]:
        raise SystemExit(f"theta wells_dir {meta.get('wells_dir')!r} is not "
                         f"{C.SPEC['heads']['wells_dir']!r}")
    data_dir = str(C.repo_path(C.SPEC["data_src"]))
    never = out.parent / "_dump_fit_heads_never_written"
    cc_argv = hindcast_argv(data_dir, str(never))
    heads, inp, call = capture(cc, cc_argv)
    if never.exists():
        raise SystemExit(f"{never} was created: main ran past the hindcast")
    if not (call["gain"] == 1.0 and call["every"] == 1 and call["apex_from_given"]
            and call["update_method"] == "innovation"):
        raise SystemExit(f"the captured hindcast call is not the declared nudging: {call}")
    keep = inp.dates.to_period("M") <= pd.Period(C.SPEC["heads"]["last_month"], freq="M")
    n = int(np.asarray(keep).sum())
    g = inp.grid
    prov = {"what": "calibrate_coupled.main's hindcast_with_nudging result (captured)",
            "argv": cc_argv, "call": call, "theta": C.SPEC["theta"],
            "theta_sha256": C.file_sha256(theta), "wells_dir": meta.get("wells_dir"),
            "record": [str(inp.dates[0].date()), str(inp.dates[-1].date())],
            "written_months": [str(inp.dates[0].date()), str(inp.dates[n - 1].date())],
            "dtype": str(heads.dtype), "shape_written": [1, *heads.shape[:2], n],
            "spec_sha256": C.spec_sha256(), "code": {k: v for k, v in C.code_state().items()
                                                     if k != "files"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as fh:
        np.savez_compressed(
            fh, heads_mean=heads[None, :, :, :n],
            dates=np.array([str(d.date()) for d in inp.dates[:n]]),
            scenario_names=np.array(["baseline"]), mask=np.asarray(g.mask, dtype=bool),
            x0=float(g.x0), y0=float(g.y0), dx=float(g.dx),
            provenance_json=np.array(json.dumps(prov)))
    print(f"wrote {out}: {heads.shape[0]} layers, {heads.shape[1]} cells, {n} of "
          f"{heads.shape[2]} months (<= {C.SPEC['heads']['last_month']}), {heads.dtype}; "
          f"sha256 {C.file_sha256(out)}; code tree {prov['code']['tree_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
