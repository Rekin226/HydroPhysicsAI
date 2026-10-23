"""Guard probe (U3/U4/U5) of the P2def products after publication: every use of the vocabulary
is opened through ``common.guard.open_product``; the expected outcome follows from the design
(allowed uses; a failed gate leaves only design_time_evidence and after_the_fact_analysis;
pre-origin uses of the pre-origin products pass only through the declared exemptions; the
post-origin products refuse every pre-origin use).

Usage: python -I r_guard_probe.py <out_json>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import r_ledger as RL  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.uses import GATE_EXEMPT_USES, USES  # noqa: E402


def main(out: str) -> None:
    res, n, n_ok = [], 0, 0
    for prod in ("census_post2016/preorigin", "census_post2016/full", "census_extension/full",
                 "layer_response"):
        pv = RL.latest(prod)
        d = D.DATAPREP / pv
        m, _ = manifest.load_manifest(d)
        gate = bool(m["gate"]["passed"])
        for u in sorted(USES):
            exp = u in m["allowed_uses"] and (gate or u in GATE_EXEMPT_USES)
            try:
                guard.open_product(d, u)
                got, why = True, ""
            except Exception as e:  # noqa: BLE001 - the probe records any refusal
                got, why = False, f"{type(e).__name__}: {str(e)[:140]}"
            n += 1
            n_ok += got == exp
            res.append({"product": pv, "use": u, "opened": got, "expected": exp,
                        "as_designed": got == exp, "why": why})
    # time-sliced forcing read: flags 0-5 only
    pv = RL.latest("census_post2016/preorigin")
    try:
        p = guard.open_product(D.DATAPREP / pv, "after_the_fact_analysis")
        t = p.read_table("county_class_monthly.parquet")
        res.append({"product": pv, "use": "read_table after_the_fact_analysis",
                    "flags": sorted(int(x) for x in t["flag"].unique()), "rows": int(len(t))})
    except Exception as e:  # noqa: BLE001
        res.append({"product": pv, "use": "read_table", "error": str(e)[:200]})
    D.save_json({"n": n, "as_designed": n_ok, "probes": res}, out)
    print(f"{n_ok}/{n} probes as designed")
    for r in res:
        if not r.get("as_designed", True):
            print("NOT AS DESIGNED", r)


if __name__ == "__main__":
    main(sys.argv[1])
