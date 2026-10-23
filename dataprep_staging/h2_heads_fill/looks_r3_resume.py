"""Look-log disclosure of the interrupted first attempt of WP H2 repair round 3 (plan 4.5).

The first attempt registered ``H2.repair3.v1`` (2026-10-13T20:54:25Z), generated family-d
realizations r = 60..68 (20:54:52Z..20:57:22Z) and started the registered ``H2.ic.twin``
scoring (look logged by ``assert_can_score`` at 20:54:33Z). A usage limit stopped the session
before that scoring wrote or printed anything and before r = 69 or ``H2.fresh3.d`` existed.
The resumed session (2026-10-14T00:04Z..) verified the 47 registered input hashes, generated
r = 69, scored ``H2.fresh3.d`` and re-ran ``H2.ic.twin`` under the same registration (second
scoring look 00:05:03Z; the computation is deterministic and no earlier output existed). This
record makes that history explicit; it is written after both scorings.

Usage: python -I looks_r3_resume.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402


def main():
    rec = prereg3.record_look(
        ["H2:fresh3_d_r60-69", "H2:ic_twin_jan2013-2022"], "heads:2012-01..2022-12", C.WP,
        "resume_disclosure", prereg_id=R3.PREREG_ID_R3,
        note="Resume disclosure (H2 repair round 3): the first H2.ic.twin scoring "
             "(assert_can_score look 2026-10-13T20:54:33Z) was cut off by a usage limit before "
             "it wrote or printed any result; family d r = 60..68 had been generated, r = 69 "
             "and H2.fresh3.d had not. Resumed 2026-10-14T00:04Z: registered inputs verified "
             "(47/47 unchanged), r = 69 generated, H2.fresh3.d scored once (00:05:03Z), "
             "H2.ic.twin re-run once under the same registration (00:05:03Z). A dry run of "
             "the full-variant bridge packaging (no comparison with truth) and a row-identity "
             "check against heads/L2_filled_full/v1 (flags 0-2, after_the_fact_analysis) "
             "followed.")
    print(json.dumps({k: rec[k] for k in ("time", "wp", "purpose", "units")}, indent=1))


if __name__ == "__main__":
    main()
