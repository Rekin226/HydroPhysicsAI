"""WP S5 look log, third session (2026-10-17): looks outside a registered score.

The second session (2026-10-14 evening) finished procedure H, scored T2, T3 and R2, and was cut
off by the usage limit while ``s5hyper.py t1`` was in its stage-A re-fit (look
2026-10-14T16:35:24Z; its log S5_evidence/hyper_t1.log printed restricted marginal-likelihood
values only and wrote no hyper_t1.json). This module logs the looks of the third session BEFORE
each step runs:

    python -I s5looks3.py hyper_t1_restart   T1 stage A restarted from scratch (same code)
    python -I s5looks3.py product_build      full-data fit for subsidence_fused/v1
    python -I s5looks3.py diagnostics        after-the-fact diagnostics of the report

Registered units carry their members in the pre-registration; no new unit is declared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
from harness import prereg3  # noqa: E402

PRE_UNITS = ["s5.cmp.preorigin", "s5.gnss.preorigin", "s5.insar.runP", "s5.lev.t2_blocks"]
ALL_UNITS = PRE_UNITS + ["s5.post2022.families"]

LOOKS = {
    "hyper_t1_restart": dict(
        units=PRE_UNITS, scope="subsidence_preorigin:2012-01..2022-12",
        purpose="hyperparameter_fit",
        note=("registered T1 stage A (s5hyper.py t1) restarted from scratch with the registered, "
              "hash-verified code: the previous run (look 2026-10-14T16:35:24Z) was cut off by "
              "the usage limit in the stage-A re-fit; its log printed restricted "
              "marginal-likelihood values only and no output was written. The withheld "
              "2020-2021 surveys do not enter. No score is computed.")),
    "product_build": dict(
        units=ALL_UNITS, scope="subsidence_all:2012-01..2026-01", purpose="product_build",
        note=("subsidence_fused/v1 build (s5product.py): full-data fit with the pre-origin "
              "hyperparameters and depth fractions; GNSS and ring/pile screening by one |z| > 5 "
              "pass on the full fit; pre-origin variant fit with the same code on pre-origin "
              "data. Nothing here enters a gate, threshold, hyperparameter or member choice "
              "(U7). Run after T1, T2, T3 and R2 were scored.")),
    "diagnostics": dict(
        units=ALL_UNITS, scope="subsidence_all:2012-01..2026-01", purpose="after_the_fact",
        note=("after-the-fact diagnostics for the S5 report (s5diag.py), run after every "
              "registered test was scored: posterior correlations of the depth fractions with "
              "the field, T1/T2 error breakdowns, VR-mask summaries, T2 coverage by interval. "
              "Report only; nothing feeds back into the product, a gate or a threshold.")),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=sorted(LOOKS))
    a = ap.parse_args(argv)
    lk = LOOKS[a.what]
    rec = prereg3.record_look(lk["units"], lk["scope"], C.WP, lk["purpose"],
                              prereg_id=C.PREREG_ID, note=lk["note"])
    print(json.dumps({k: rec[k] for k in ("time", "purpose", "units", "prereg_id")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
