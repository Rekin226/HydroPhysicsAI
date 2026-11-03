"""FB.family_power.v1 runner (round-8 rank 16, FB part): CPU only, 0 GPU-h.

    python -I dataprep_staging/fb_family/power_fb.py smoke --seed <int> --out <dir outside results/>
        SYNTHETIC inputs only (fb_lib.synthetic_u1 / synthetic_inputs); sets nothing; refuses
        under results/ and refuses the registered seed; every smoke seed is logged in
        <out>/attempts.jsonl (and the builder's are listed in BUILDER_SEEDS).
    python -I dataprep_staging/fb_family/power_fb.py run --seed <the registered seed>
        The registered study, once (the output directory is fixed; tests pass another). Refuses
        a refused seed (BUILDER_SEEDS, U1's and FV's seeds), and unless FB.family_power.v1 is
        in INDEX.jsonl with its file sha256, its 'design' equals design_spec(U1's
        registration, the input sha256, seed), its code_sha256 equals the current files, U1's
        and FV's bound code is unchanged and result.json does not exist. An engineering
        failure (input identity, sigma_idio units/lead gate, forward metadata, self-check)
        writes no result. Every invocation, a crash included, is appended to
        results/dataprep/reports/FB.family_power.v1/attempts.jsonl.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "fv_decidability"),
           str(HERE.parent / "e02c")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fb_inputs as FI  # noqa: E402
import fb_io as io  # noqa: E402
import fb_lib as L  # noqa: E402

CODE_FILES = (f"{io.X}/fb_lib.py", f"{io.X}/fb_io.py", f"{io.X}/fb_inputs.py",
              f"{io.X}/power_fb.py", "dataprep_staging/e02d/e02dlib.py",
              "dataprep_staging/e02d/power_e02d.py", "dataprep_staging/e02d/e02dio.py",
              "dataprep_staging/fv_decidability/osse_fv.py",
              "dataprep_staging/e02c/score_e02c.py")
U1_BOUND = ("dataprep_staging/e02d/power_e02d.py", "dataprep_staging/e02d/e02dlib.py",
            "dataprep_staging/e02d/e02dio.py")
FV_BOUND = ("dataprep_staging/fv_decidability/osse_fv.py",)
RESULT_DIR = f"results/dataprep/reports/{L.POWER_ID}"
ATTEMPTS = f"{RESULT_DIR}/attempts.jsonl"
#: seeds the builder used on SYNTHETIC inputs (tests, smokes, timing); never registrable
BUILDER_SEEDS = (1, 3, 5, 7, 11, 13, 424242)
#: U1.decomp_power.v1 (seed, self_check_seed) and FV v1/v2 (SEED): never reused (S6)
UPSTREAM_SEEDS = (20261101, 20261103, 20261104)
REFUSED_SEEDS = tuple(sorted(set(BUILDER_SEEDS) | set(UPSTREAM_SEEDS)))


def code_sha() -> dict:
    return {f: io.sha256(io.repo_path(f)) for f in CODE_FILES}


def seed_problem(seed: int) -> str | None:
    if seed in REFUSED_SEEDS:
        return f"seed {seed} is refused (builder or upstream seed: {list(REFUSED_SEEDS)})"
    return None


def bound_problems(u1_doc: dict, fv2_doc: dict | None) -> list[str]:
    bad = [f for f in U1_BOUND
           if u1_doc.get("code_sha256", {}).get(f) != io.sha256(io.repo_path(f))]
    if fv2_doc is not None:
        bad += [f for f in FV_BOUND
                if fv2_doc.get("frozen_code", {}).get(f) != io.sha256(io.repo_path(f))]
    return [f"bound code changed since its registration: {f}" for f in bad]


def design_now(seed: int, prereg_dir=None) -> dict:
    u1, u1_sha = io.registered(L.U1_ID, prereg_dir)
    return L.design_spec(u1, u1_sha, FI.input_sha(), seed)


def run_registered(seed: int, prereg_dir=None, out: Path | None = None, log=print) -> int:
    """The registered run. ``out`` exists for the test suite only (the CLI has no --out)."""
    import osse_fv as FV

    out = Path(out) if out else io.repo_path(RESULT_DIR)
    att = out / "attempts.jsonl"
    rec = {"cmd": "run", "seed": seed}
    t0 = time.time()
    try:
        why = seed_problem(seed)
        if why:
            raise io.Refused(why)
        doc, sha = io.registered(L.POWER_ID, prereg_dir)
        rec["declaration_sha256"] = sha
        if doc.get("design", {}).get("seed") != seed:
            raise io.Refused(f"seed {seed} is not the registered seed")
        if (out / "result.json").exists():
            raise io.Refused("result.json exists (the study runs once)")
        design = design_now(seed, prereg_dir)
        if doc.get("design") != design:
            raise io.Refused("the registered design differs from design_spec(U1, input sha256, "
                             "seed): changed inputs or registration")
        if doc.get("code_sha256") != code_sha():
            raise io.Refused("code changed since registration")
        u1, _ = io.registered(L.U1_ID, prereg_dir)
        fv2, _ = io.registered(L.FV2_ID, prereg_dir)
        bad = bound_problems(u1, fv2)
        if bad:
            raise io.Refused("; ".join(bad))
        inputs, prov = FI.load_real(design)
        res = L.run(design, inputs, seed, design["replicates"], design["cal_replicates"],
                    smoke=False, log=log)
    except (io.EngineeringFailure, L.EngineeringFailure, FV.EngineeringFailure) as exc:
        io.append_attempt(att, {**rec, "outcome": "engineering_failure",
                                "why": f"{type(exc).__name__}: {exc}"})
        print(f"ENGINEERING FAILURE (nothing written): {exc}")
        return 2
    except io.Refused as exc:
        io.append_attempt(att, {**rec, "outcome": "refused", "why": str(exc)})
        print(f"REFUSED: {exc}")
        return 1
    except Exception as exc:                      # noqa: BLE001 - log every crash
        io.append_attempt(att, {**rec, "outcome": "crashed",
                                "why": f"{type(exc).__name__}: {exc}",
                                "traceback_tail": traceback.format_exc().splitlines()[-6:]})
        print(f"CRASHED (nothing written): {type(exc).__name__}: {exc}")
        return 3
    res["declaration"] = {"id": L.POWER_ID, "sha256": sha}
    res["design_sha256"] = io.sha_json(design)
    res["inputs_provenance"] = prov
    res["code_sha256"] = code_sha()
    res["runtime_s"] = round(time.time() - t0, 1)
    res["created"] = io.utc_stamp()
    h = io.write_once(out / "result.json", res)
    io.append_attempt(att, {**rec, "outcome": "written", "result_sha256": h,
                            "registrable": res["registration"]["registrable"]})
    print(f"wrote {out / 'result.json'} sha256 {h}")
    print("registrable:", res["registration"]["registrable"])
    return 0


def smoke(seed: int, out: Path, prereg_dir=None, reps: int | None = None,
          cal: int | None = None, log=print) -> Path:
    out = Path(out).resolve()
    res_root = io.repo_path("results").resolve()
    if out == res_root or res_root in out.parents:
        raise io.Refused("a smoke run never writes under results/")
    if io.index_record(L.POWER_ID, prereg_dir) is not None:
        doc, _ = io.registered(L.POWER_ID, prereg_dir)
        if doc.get("design", {}).get("seed") == seed:
            raise io.Refused("the registered seed is never used on synthetic inputs")
    io.append_attempt(out / "attempts.jsonl", {"cmd": "smoke", "seed": seed,
                                               "synthetic": True})
    design = L.design_spec(L.synthetic_u1(seed), "synthetic", {"synthetic": "synthetic"},
                           seed)
    res = L.run(design, L.synthetic_inputs(design, seed), seed,
                reps or L.SMOKE["replicates"], cal or L.SMOKE["cal_replicates"], smoke=True,
                log=log)
    res["synthetic"] = True
    path = out / "smoke_result.json"
    io.write_once(path, res)
    return path


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="SYNTHETIC inputs only; sets nothing")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="the registered study (refuses unless registered)")
    r.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        try:
            p = smoke(a.seed, a.out, prereg_dir)
        except io.Refused as exc:
            print(f"REFUSED: {exc}")
            return 1
        print(f"wrote {p} (synthetic; sets nothing)")
        return 0
    return run_registered(a.seed, prereg_dir)


if __name__ == "__main__":
    raise SystemExit(main())
