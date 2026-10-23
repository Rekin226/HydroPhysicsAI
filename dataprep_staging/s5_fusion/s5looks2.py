"""WP S5 look log, resumed session (2026-10-14 evening): looks outside a registered score.

The first session registered ``S5.subsidence_fusion.v1`` (02:08:05Z), logged the start of
procedure H (02:08:15Z) and was cut off by the usage limit during stage A (evaluation 180 of
220; its log printed only restricted marginal-likelihood values, no residual or score). This
module logs the looks of the resumed session BEFORE each step runs:

    python -I s5looks2.py hyper_restart     procedure H restarted from scratch (same code)
    python -I s5looks2.py hyper_t1          T1 stage A on the training leveling (<= 2019-12)
    python -I s5looks2.py product_build     full-data fit for subsidence_fused/v<N>
    python -I s5looks2.py diagnostics       after-the-fact diagnostics of the report

Registered units of the pre-registration carry their members there; no new unit is declared.
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
    "hyper_restart": dict(
        units=PRE_UNITS, scope="subsidence_preorigin:2012-01..2022-12",
        purpose="hyperparameter_fit",
        note=("registered procedure H (s5hyper.py main) restarted from scratch with the "
              "registered, hash-verified code: the first run (look 2026-10-14T02:08:15Z) was cut "
              "off by the usage limit at stage A evaluation 180 of 220; its log "
              "(S5_evidence/hyper_pre.log) printed restricted marginal-likelihood values only. "
              "No residual, prediction or score is computed by procedure H.")),
    "hyper_t1": dict(
        units=PRE_UNITS, scope="subsidence_preorigin:2012-01..2022-12",
        purpose="hyperparameter_fit",
        note=("registered T1 stage A (s5hyper.py t1): leveling hyperparameters re-estimated on "
              "the T1 training leveling (surveys <= 2019-12-31, main screening flags applied); "
              "the withheld 2020-2021 surveys do not enter. No score is computed.")),
    "product_build": dict(
        units=ALL_UNITS, scope="subsidence_all:2012-01..2026-01", purpose="product_build",
        note=("subsidence_fused/v<N> build (s5product.py): full-data fit (leveling 2012-2021, "
              "gnss_clean/full/v2, insar_069A/v2 run F, rings L0 to 2025-05, printed piles to "
              "2025-11) with the pre-origin hyperparameters and depth fractions; GNSS and "
              "ring/pile screening by one |z| > 5 pass on the full fit. Nothing here enters a "
              "gate, threshold, hyperparameter or member choice (U7).")),
    "diagnostics": dict(
        units=ALL_UNITS, scope="subsidence_all:2012-01..2026-01", purpose="after_the_fact",
        note=("after-the-fact diagnostics for the S5 report (s5diag.py), run after every "
              "registered test was scored: posterior correlations of the depth fractions with "
              "the field, T1/T2 error breakdowns by region and interval, VR-mask summaries. "
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
