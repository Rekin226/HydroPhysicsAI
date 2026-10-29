"""Emit run_e2.sh's job list and argv from the frozen spec, so the batch cannot drift from it.

    e2flags.py jobs SYNTH_DIR             one line per arm: ARM|TRUTH|FORM|ORIGIN|DATA_DIR
    e2flags.py argv ARM SYNTH_DIR RUNS    calibrate_coupled's argv of ARM, one item per line

DATA_DIR is the generator manifest's synthetic data dir of the arm's truth (before the
generator has run, a dry run prints where it will be).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as C  # noqa: E402


def data_dirs(synth: Path) -> dict[str, str]:
    mp = synth / "generator_manifest.json"
    if mp.exists():
        return {t: v["data_dir"] for t, v in json.loads(mp.read_text())["truths"].items()}
    return {t: str(synth.resolve() / f"truth_{t}" / "data") for t in C.SPEC["primary_truths"]}


def jobs(synth: Path) -> list[str]:
    dd = data_dirs(synth)
    return ["|".join([a, t, f, o, dd[t]]) for a, t, f, o in C.all_arms()]


def argv(arm: str, synth: Path, runs: Path) -> list[str]:
    for a, t, f, o in C.all_arms():
        if a == arm:
            return C.fit_args(f, o, data_dirs(synth)[t], str(runs.resolve() / a))
    raise SystemExit(f"{arm!r} is not a declared arm")


def main(args: list[str]) -> int:
    if args[:1] == ["jobs"]:
        print("\n".join(jobs(Path(args[1]))))
        return 0
    if args[:1] == ["argv"]:
        print("\n".join(argv(args[1], Path(args[2]), Path(args[3]))))
        return 0
    raise SystemExit(f"usage: e2flags.py jobs SYNTH | argv ARM SYNTH RUNS (got {args!r})")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
