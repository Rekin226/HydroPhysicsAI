"""WP S3 repair round 2026-10-13: v1 against v2 checks for each verifier finding (after
``v2/build.py`` for both variants).

    python -I dataprep_staging/s3_gnss/v2/repair_checks.py OUT_JSON [--ledger]

Reads only the published product directories (v1 and v2 of both variants) and their manifests:
station sets and roles, steps removed by the re-test and detected steps left below delta_min,
flag shares and the flag-3 days of FRES, ANES flags in the full variant, the wra_hf window,
manifest exemptions and the v1 provenance record. With ``--ledger`` appends one ledger row per
number and source to ``results/dataprep/ledger/rows/S3.csv``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import ledger  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

VARIANTS = ("preorigin", "full")


def load(variant: str, version: str) -> dict:
    d = DATAPREP / "gnss_clean" / variant / version
    out = {"dir": d, "manifest": json.loads((d / "MANIFEST.json").read_text()),
           "stations": pd.read_csv(d / "stations.csv"), "steps": pd.read_csv(d / "steps.csv"),
           "wra": pd.read_parquet(d / "wra" / "daily.parquet",
                                  columns=["station", "date", "flag", "method"]),
           "ngl": pd.read_parquet(d / "ngl" / "daily.parquet",
                                  columns=["station", "date", "flag", "method"])}
    return out


def main(argv: list[str]) -> int:
    res: dict = {}
    rows: list[dict] = []

    def num(nid, value, srcs, pv, note=""):
        res[nid] = value
        for s in srcs:
            rows.append({"number_id": nid, "source": s, "role": "input", "product_version": pv,
                         "note": note})

    for var in VARIANTS:
        v1, v2 = load(var, "v1"), load(var, "v2")
        s1, s2 = rel_to_repo(v1["dir"]), rel_to_repo(v2["dir"])
        pv = f"gnss_clean/{var}/v2"
        for fam in ("wra", "ngl"):
            a = set(v1["stations"][(v1["stations"].family == fam)
                                   & (v1["stations"].role != "dropped")].station)
            b = set(v2["stations"][(v2["stations"].family == fam)
                                   & (v2["stations"].role != "dropped")].station)
            num(f"S3.repair.{var}.{fam}.stations_v1", sorted(a), [s1], pv)
            num(f"S3.repair.{var}.{fam}.stations_v2", sorted(b), [s2], pv)
            num(f"S3.repair.{var}.{fam}.added_in_v2", sorted(b - a), [s1, s2], pv)
            num(f"S3.repair.{var}.{fam}.removed_in_v2", sorted(a - b), [s1, s2], pv)
            roles = v2["stations"][v2["stations"].family == fam].set_index("station").role
            num(f"S3.repair.{var}.{fam}.roles_v2", roles.to_dict(), [s2], pv)
            for ver, t, src in (("v1", v1[fam], s1), ("v2", v2[fam], s2)):
                sh = {int(k): round(float(x), 4) for k, x in
                      t.flag.value_counts(normalize=True).sort_index().items()}
                num(f"S3.repair.{var}.{fam}.flag_shares_{ver}", sh, [src], pv)
        # steps
        dmin = v2["manifest"]["parameters"]["detector"]["delta_min"]
        for ver, v, src in (("v1", v1, s1), ("v2", v2, s2)):
            st = v["steps"]
            det = st[(st.evidence == "detected") & (st.size_m.abs() < dmin)]
            num(f"S3.repair.{var}.steps_{ver}", int(len(st)), [src], pv)
            num(f"S3.repair.{var}.detected_below_delta_min_{ver}", int(len(det)), [src], pv,
                f"delta_min {dmin}")
            w = st[st.family == "wra"]
            yr = pd.to_datetime(w.date).dt.year
            num(f"S3.repair.{var}.wra_steps_{ver}", {"n": int(len(w)),
                                                    "2020-2022": int((yr <= 2022).sum()),
                                                    "2023-2025": int((yr >= 2023).sum()),
                                                    "flag3_basis": int((w.flag_after == 3).sum())},
                [src], pv)
        notes = v2["stations"].get("notes", pd.Series(dtype=str)).fillna("")
        dropped = [f"{r.station}: {x}" for r, s_ in zip(v2["stations"].itertuples(), notes,
                                                        strict=True)
                   for x in s_.split(" | ") if "re-test" in x]
        num(f"S3.repair.{var}.retest_dropped", dropped, [s2], pv)
        # FRES flag-3 days (the verifier's example)
        for ver, v, src in (("v1", v1, s1), ("v2", v2, s2)):
            f = v["wra"][v["wra"].station == "FRES"]
            num(f"S3.repair.{var}.FRES_flag3_days_{ver}", int((f.flag == 3).sum()), [src], pv)
        # ANES
        a = v2["wra"][v2["wra"].station == "ANES"]
        num(f"S3.repair.{var}.ANES_v2", {"days": int(len(a)),
                                         "flags": {int(k): int(n) for k, n in
                                                   a.flag.value_counts().sort_index().items()},
                                         "jump_cluster_days": int((a.method == "jump_cluster")
                                                                  .sum()),
                                         "jump_cluster_first": str(a[a.method ==
                                                                     "jump_cluster"].date.min())
                                         if (a.method == "jump_cluster").any() else None},
            [s2], pv)
        jc = v2["wra"][v2["wra"].method == "jump_cluster"]
        num(f"S3.repair.{var}.jump_cluster_days_by_station",
            jc.groupby("station").size().to_dict(), [s2], pv)
        # manifests
        m1, m2 = v1["manifest"], v2["manifest"]
        num(f"S3.repair.{var}.exemptions_v2", m2.get("exemptions"), [s2], pv)
        num(f"S3.repair.{var}.hp_window", {"v1": m1["parameters"]["detector"]["hp_window"],
                                           "v2": m2["parameters"]["detector"]["hp_window"]},
            [s1, s2], pv)
        num(f"S3.repair.{var}.v2_supersedes_record",
            m2["parameters"].get("supersedes"), [s2], pv)
        num(f"S3.repair.{var}.catalogue_note_v2", m2["parameters"].get("wra_catalogue_note"),
            [s2], pv)
        num(f"S3.repair.{var}.gate_v2", m2.get("gate"), [s2], pv)
        num(f"S3.repair.{var}.reference_v2", m2["parameters"].get("reference_stations"),
            [s2], pv)
    Path(argv[0]).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False, default=str)[:400])
    if "--ledger" in argv:
        p = ledger.append_rows("S3", rows)
        print(f"ledger: {len(rows)} rows -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
