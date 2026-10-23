"""WP A0 repair round: pre-origin check of the partial-month error rho(k) used by plan U2.

Plan U2 sets sigma_obs = max(0.10 m, rho(k)) from the gap-fill pilot (0.33 m at k = 20, 0.49
at k = 15, 0.64 at k = 10). The data-window audit found that s4b_partial.py computed rho(k)
from complete months of 2012-2026, i.e. including the 2023-26 test heads; a weight of the
calibration likelihood is a hyperparameter, which U7 says no 2023+ observation may set.

This script reruns the archived s4b_partial.py twice from hash-checked copies, on the
archived daily_clean.parquet:
  full        unmodified (must reproduce out/partial_month_error.csv)
  preorigin   one inserted line keeping days before 2023-01-01 only
and records both tables. Design-time evidence for WP H2/A1/I1; not a gate.

Usage (through tools/cpu_slot.sh): python -m common.a0.r2_rho_k
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from common.a0.rerun_pilots import PY, compare_csv, copy_script, link, run  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PILOTS, REPORTS_DIR, rel_to_repo  # noqa: E402

S = "head_gapfill_pilot"
BASE = REPORTS_DIR / "A0_evidence" / "repair" / "reruns" / "rho_k"
ANCHOR = "D = D[D.sid.isin(set(sids))].copy()"
PATCH = (ANCHOR, ANCHOR + '\nD = D[D.date < "2023-01-01"].copy()  # A0 repair: pre-origin only')


def one(name: str, patch) -> dict:
    d = BASE / name
    d.mkdir(parents=True)
    (d / "out").mkdir()
    rec = {"variant": name, "dir": rel_to_repo(d),
           "scripts": [copy_script(S, "lib.py", d), copy_script(S, "s4b_partial.py", d, patch)]}
    for f in ("daily_clean.parquet", "monthly_long.parquet", "well_meta.csv"):
        link(PILOTS / S / "out" / f, d / "out" / f)
    rec["step"] = run([PY, "s4b_partial.py"], d, d / "s4b.log")
    rec["table"] = pd.read_csv(d / "out" / "partial_month_error.csv").to_dict("records")
    return rec


def main() -> int:
    if BASE.exists():
        raise SystemExit(f"{BASE} exists; never overwritten")
    full = one("full", None)
    pre = one("preorigin", PATCH)
    ref = PILOTS / S / "out" / "partial_month_error.csv"
    full["reproduces_archive"] = compare_csv(ref, BASE / "full" / "out" /
                                             "partial_month_error.csv")
    a = pd.read_csv(ref).set_index("k")
    b = pd.read_csv(BASE / "preorigin" / "out" / "partial_month_error.csv").set_index("k")
    cmp = pd.DataFrame({"n_2012_2026": a.n, "rmse_2012_2026": a.rmse, "n_2012_2022": b.n,
                        "rmse_2012_2022": b.rmse})
    cmp["rmse_rel_change"] = cmp.rmse_2012_2022 / cmp.rmse_2012_2026 - 1
    res = {"archive": {"path": rel_to_repo(ref), "sha256": sha256_file(ref)},
           "full": full, "preorigin": pre, "comparison": cmp.reset_index().to_dict("records"),
           "plan_u2_values": {"20": 0.33, "15": 0.49, "10": 0.64}}
    (BASE.parent / "rho_k_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n")
    print(cmp.round(4).to_string())
    print("full reproduces archive:", full["reproduces_archive"]["ok"],
          full["reproduces_archive"]["identical_bytes"])
    return 0 if full["step"]["rc"] == 0 and pre["step"]["rc"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
