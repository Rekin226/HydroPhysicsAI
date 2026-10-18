"""WP S3 repair round 2026-10-13: ledger rows for the report numbers not written by
``report_v2.py`` or ``repair_checks.py`` (guard probes, checker, the A2 pair diagnostic, the
tune rerun check, the look log and the pre-registration disclosures).

    python -I dataprep_staging/s3_gnss/v2/repair_rows.py EVIDENCE_DIR [--ledger]

Every value is recomputed from the files named as sources; nothing is typed in.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import ledger  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, rel_to_repo  # noqa: E402

PRE1 = DATAPREP / "gnss_clean" / "preorigin" / "v1"
PRE2 = DATAPREP / "gnss_clean" / "preorigin" / "v2"
FULL2 = DATAPREP / "gnss_clean" / "full" / "v2"
LOOKS = PREREG_DIR / "looks.jsonl"
PREREG = PREREG_DIR / "S3.gnss_clean_acceptance.v2.json"
OLD_TUNE_LOG = DATAPREP / "reports" / "S3_evidence" / "repair_r3" / "tune_r3" / "tune.log"


def main(argv: list[str]) -> int:
    ev = Path(argv[0])
    res: dict = {}
    rows: list[dict] = []

    def num(nid, value, srcs, pv, note=""):
        res[nid] = value
        for s, role in srcs:
            rows.append({"number_id": nid, "source": rel_to_repo(s) if isinstance(s, Path)
                         else s, "role": role, "product_version": pv, "note": note})

    # guard probes
    g = json.loads((ev / "guard_probe_v2.json").read_text())
    out = {f"{r['product']}:{r['use']}": ("ok" if r["ok"] else r["error"].split(":")[0])
           for r in g}
    num("S3.repair.guard_probes_v2", out, [(ev / "guard_probe_v2.json", "input"),
                                           (PRE2, "input"), (FULL2, "input")],
        "gnss_clean/{preorigin,full}/v2", f"{len(g)} probes")
    # checker
    c = json.loads((ev / "checker_gnss_clean_v1_v2.json").read_text())
    summ = dict(c["summary"])
    summ["n_orphans_listed"] = len(c.get("orphans") or [])
    num("S3.repair.checker", summ, [(ev / "checker_gnss_clean_v1_v2.json", "input"),
                                    (DATAPREP / "gnss_clean", "input"),
                                    ("data_fetch_api/new_measurements/ngl_gnss", "input")],
        "gnss_clean v1+v2; L0 ngl_gnss")
    # A2 pair diagnostic: steps and Jan/Dec flags of the stations whose evaluability changed
    for ver, d in (("v1", PRE1), ("v2", PRE2)):
        st = pd.read_csv(d / "steps.csv")
        wd = pd.read_parquet(d / "wra" / "daily.parquet", columns=["station", "date", "flag"])
        for k in ("GFES", "TKJS", "YCES", "KTES"):
            s = st[(st.family == "wra") & (st.station == k)]
            x = wd[wd.station == k].assign(m=lambda t: t.date.str[:7])
            x = x[x.m.isin(["2020-01", "2020-12", "2021-01", "2021-12"])]
            num(f"S3.repair.a2diag.{ver}.{k}", {
                "steps": [[r.date, round(1000 * r.size_m, 1), r.evidence, r.basis,
                           int(r.flag_after)] for r in s.itertuples()],
                "jan_dec_flag01_days": {m: int(g_.flag.isin([0, 1]).sum())
                                        for m, g_ in x.groupby("m")}},
                [(d, "input")], f"gnss_clean/preorigin/{ver}",
                "why the evaluable A2 pairs changed (diagnose look 2026-10-13T08:15Z)")
    # tune rerun reproduces the cut-off run's first 12 configurations
    new = (ev / "tune_r3" / "tune.log").read_text().splitlines()
    old = OLD_TUNE_LOG.read_text().splitlines()

    def cfg(lines):
        return [re.sub(r" \(\d+s\)$", "", x) for x in lines if x.startswith('{"delta_min"')]
    a, b = cfg(new), cfg(old)
    num("S3.repair.tune_r3.rerun_matches_cutoff", {"cutoff_configs": len(b),
                                                   "identical": a[:len(b)] == b,
                                                   "rerun_configs": len(a)},
        [(ev / "tune_r3" / "tune.log", "input"), (OLD_TUNE_LOG, "input")],
        "gnss_clean/design (pre-origin tune r3)")
    # looks of this round
    lk = [json.loads(x) for x in LOOKS.read_text().splitlines() if x.strip()]
    mine = [x for x in lk if x["wp"] == "S3" and x["logged"] >= "2026-10-13"]
    num("S3.repair.looks_this_round", [[x["logged"][:19], x["purpose"], x.get("prereg_id"),
                                        x["data_scope"]] for x in mine],
        [(LOOKS, "input")], "n/a", f"{len(mine)} looks, harness.prereg3")
    v1rep = [x for x in lk if x["wp"] == "S3" and "independent verifier" in x.get("note", "")]
    num("S3.repair.verifier_looks", [[x["logged"][:19], x["purpose"], x["note"][:160]]
                                     for x in v1rep], [(LOOKS, "input")], "n/a",
        "verifier's re-runs and LNJS/YSLL/ANES statistics as recorded in the look log")
    # pre-registration
    p = json.loads(PREREG.read_text())
    num("S3.repair.prereg_v2", {"created": p["created"], "tool": p["tool"],
                                "n_inputs": len(p["inputs"]),
                                "unverifiable_prior_looks_listed":
                                    len(p.get("unverifiable_prior_looks") or [])},
        [(PREREG, "input")], "prereg")
    vp = json.loads((ev / "acceptance" / "validation_preorigin.json").read_text())
    num("S3.repair.assert_can_score", vp["looks_logged"][0],
        [(ev / "acceptance" / "validation_preorigin.json", "input")], "gnss_clean/preorigin/v2")
    (ev / "repair_rows.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                    default=str))
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False, default=str)[:300])
    if "--ledger" in argv:
        path = ledger.append_rows("S3", rows)
        print(f"ledger: {len(rows)} rows -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
