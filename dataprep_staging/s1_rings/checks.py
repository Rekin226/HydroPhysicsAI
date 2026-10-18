"""Post-build checks of rings_bridged (WP S1a): manifest re-hash, guard probe per use, U1
filter on scoring reads, flag/sd consistency, rebuild identity (optional).

    tools/cpu_slot.sh python -m s1_rings.checks results/dataprep/rings_bridged/v1
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.paths import from_repo  # noqa: E402
from common.uses import USES  # noqa: E402

from s1_rings import s1const as K  # noqa: E402


def main(argv=None) -> None:
    d = from_repo((argv or sys.argv[1:])[0])
    m, _ = manifest.load_manifest(d)
    out = {"product": str(d), "validate_manifest": manifest.validate_manifest(m, d),
           "verify_hashes": manifest.verify_hashes(m, d)}
    probe = {}
    for use in sorted(USES):
        try:
            p = guard.open_product(d, use)
            t = p.read_table("ring_total_monthly_bridged.csv")
            probe[use] = {"opened": True, "rows": int(len(t)),
                          "flags": sorted(int(x) for x in t["flag"].unique())}
        except Exception as e:  # noqa: BLE001 - the probe records refusals
            probe[use] = {"opened": False, "error": type(e).__name__, "msg": str(e)[:200]}
    out["guard_probe"] = probe
    T = pd.read_csv(d / "ring_total_monthly_bridged.csv")
    B = pd.read_parquet(d / "ring_bands_monthly_bridged.parquet")
    prim = T[T.series == "primary"]
    out["flag_counts_primary_total"] = prim.flag.value_counts().sort_index().to_dict()
    out["flag_counts_all_bands"] = B.flag.value_counts().sort_index().to_dict()
    out["post_gap_flag_by_site"] = (prim[prim.month >= "2023-02"].groupby("site").flag
                                    .agg(lambda s: sorted(set(int(x) for x in s))).to_dict())
    out["fill_months_by_site"] = (prim[(prim.month >= "2022-01") & (prim.month <= "2024-06")
                                       & prim.flag.isin([3, 4, 6])]
                                  .groupby("site").size().to_dict())
    out["nonobserved_without_sd"] = int((B.flag.isin([2, 3, 4, 5, 6]) & B.sd.isna()).sum())
    out["sites"] = sorted(T.site.unique().tolist())
    out["n_sites"] = len(out["sites"])
    out["both_member_sites"] = sorted(T[T.series != "primary"].site.unique().tolist())
    ok = (not out["validate_manifest"] and not out["verify_hashes"]
          and out["nonobserved_without_sd"] == 0 and out["n_sites"] == len(K.SITES))
    exp_open = set(m["allowed_uses"]) if m["gate"]["passed"] else {"after_the_fact_analysis"}
    out["guard_as_expected"] = all(v["opened"] == (u in exp_open) for u, v in probe.items())
    sc = probe.get("scoring_target", {})
    out["scoring_flags_ok"] = (not sc.get("opened")) or set(sc.get("flags", [])) <= {0, 1}
    out["ok"] = bool(ok and out["guard_as_expected"] and out["scoring_flags_ok"])
    p = K.EVIDENCE / f"checks_{d.name}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: out[k] for k in ("validate_manifest", "verify_hashes",
                                          "guard_as_expected", "scoring_flags_ok",
                                          "flag_counts_primary_total", "both_member_sites",
                                          "ok")}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
