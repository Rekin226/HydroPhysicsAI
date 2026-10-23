"""Run a W2W4 script under ``python -I`` (isolated mode), as the hard rules require for any
process that reads downloaded (untrusted) files.

``python -I`` drops the script's directory and the current directory from ``sys.path``, so a
planted module next to a download can never be imported. This launcher puts back only the
two code directories (this folder and ``dataprep_staging/``) and runs the named script as
``__main__`` with the remaining arguments.

Usage: python -I dataprep_staging/w2_w4_weather/isorun.py <script.py> [args...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = (HERE / sys.argv[1]).resolve()
    if script.parent != HERE or script.suffix != ".py":
        raise SystemExit(f"{script}: only scripts in {HERE} may be run")
    for p in (str(STAGING), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
