"""Guard probe (U3/U4/U5) of yunlin_electricity/{preorigin,full}/v2 after publication: every use
is opened through ``common.guard.open_product`` and the outcome recorded; ``read_table`` with
forcing_frozen on the pre-origin long table must drop flags 8/9 rows and keep 0-5.

Usage: python -I r3_guard_probe.py <out_txt>
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from common import guard  # noqa: E402
from p3lib import L  # noqa: E402

PROBES = {
    "preorigin": {"ok": ["forcing_frozen", "model_design", "prior", "prior_width_input",
                         "design_time_evidence", "validator", "after_the_fact_analysis"],
                  "refused": ["scoring_target", "baseline", "datum_fit", "calibration_target",
                              "selection", "initial_conditions", "forcing_recorded_arm"]},
    "full": {"ok": ["forcing_recorded_arm", "statistical_layer_history", "validator",
                    "after_the_fact_analysis"],
             "refused": ["forcing_frozen", "model_design", "prior", "scoring_target",
                         "calibration_target", "selection", "design_time_evidence"]},
}


def main(out_txt: str) -> None:
    lines, n_ok, n = [], 0, 0
    for v, pr in PROBES.items():
        d = L.PRODUCT_ROOT / v / "v2"
        for expect, uses in pr.items():
            for u in uses:
                n += 1
                try:
                    p = guard.open_product(d, u)
                    got = "ok"
                    why = "; ".join(x.get("input", "")[:60] for x in getattr(p, "exemptions", [])
                                    or [])
                except Exception as e:  # noqa: BLE001 - the probe records any refusal
                    got = "refused"
                    why = f"{type(e).__name__}: {str(e)[:160]}"
                good = got == expect
                n_ok += good
                lines.append(f"({v!r}, {u!r}) {got.upper()} expected {expect.upper()} "
                             f"{'as designed' if good else 'NOT AS DESIGNED'} {why}")
    p = guard.open_product(L.PRODUCT_ROOT / "preorigin" / "v2", "forcing_frozen")
    t = p.read_table("county_class_monthly.parquet")
    fl = sorted(set(int(x) for x in t["flag"].unique()))
    good = set(fl) <= {0, 1, 2, 3, 4, 5}
    n += 1
    n_ok += good
    lines.append(f"read_table(preorigin v2, forcing_frozen, county_class_monthly) flags kept {fl} "
                 f"rows {len(t)} {'as designed' if good else 'NOT AS DESIGNED'}")
    lines.append(f"{n_ok}/{n} probes as designed")
    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main(sys.argv[1])
