"""WP S3 repair round 3 (S3r): anatomy of the registered v3 results AFTER scoring (diagnosis
only; no new score, no change to the registered outcome).

    python -I dataprep_staging/s3_gnss/v3/post_test_anatomy.py EVIDENCE_DIR

1. A1, seed-707 pre-origin trials: failure classes (not found, found with |size error| > 5 mm,
   excursion or transient) by size, family and station; size-error sd of found steps; the
   share of trials whose failure is a size error only.
2. A2 second look: the v3 pre-origin pairs against the v2 ones (evaluable or not, extras), and
   the steps of TKJS, YCES, GFES and KTES dated 2019-12..2022-01 in v3 and v2 with their basis
   and flag.

Look logged beforehand: ``looks_r5.py post_test_anatomy``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.paths import DATAPREP  # noqa: E402

STATIONS = ("TKJS", "YCES", "GFES", "KTES")


def main(ev_dir: str) -> int:
    ev = Path(ev_dir)
    acc = ev / "acceptance"
    out: dict = {}
    t = pd.read_csv(acc / "injection_trials_preorigin.csv")
    t["cls"] = np.where(t.recovered, "ok", np.where(~t.found, "not_found", "size"))
    for fam, g in t.groupby("family"):
        f = g[g.found]
        rec = {"n": int(len(g)), "recovered": float(g.recovered.mean()),
               "not_found": int((g.cls == "not_found").sum()),
               "not_found_excursion": int(((g.cls == "not_found") & g.flagged_as_excursion)
                                          .sum()),
               "size_fail": int((g.cls == "size").sum()),
               "size_err_sd_mm": float(1000 * f.size_err_m.std()),
               "by_size": {f"{a * 100:.0f}cm": {"n": int(len(x)),
                                               "not_found": int((x.cls == "not_found").sum()),
                                               "size_fail": int((x.cls == "size").sum())}
                           for a, x in g.groupby(g.abs_size_m.round(3))},
               "by_station": {k: {"recovered": round(float(x.recovered.mean()), 3),
                                  "not_found": int((x.cls == "not_found").sum()),
                                  "size_fail": int((x.cls == "size").sum())}
                              for k, x in g.groupby("station")}}
        if fam == "ngl":
            rec["without_ERLN_posthoc"] = float(g[g.station != "ERLN"].recovered.mean())
        out[f"A1_{fam}"] = rec
        print(f"== {fam}: n {rec['n']} recovered {rec['recovered']:.4f}; not found "
              f"{rec['not_found']} (excursion {rec['not_found_excursion']}), size fail "
              f"{rec['size_fail']}; size err sd {rec['size_err_sd_mm']:.2f} mm")
        print("   by size:", rec["by_size"])
        worst = sorted(rec["by_station"].items(), key=lambda kv: kv[1]["recovered"])[:6]
        print("   worst stations:", worst)
        if fam == "ngl":
            print(f"   without ERLN (post-hoc subset, not a test): {rec['without_ERLN_posthoc']:.4f}")
    v3 = json.loads((acc / "validation_preorigin.json").read_text())["A2_2021_extra"]
    v2 = json.loads((ev.parent / "repair_2026-10-13" / "acceptance" /
                     "validation_preorigin.json").read_text())["A2_2021_extra"]
    cols = ["kind", "gnss", "reference", "ref_extra_mm", "uncorrected_extra_mm",
            "cm_removed_ngl_offfan_extra_mm", "cm_removed_wra_hf_extra_mm"]
    p3 = pd.DataFrame(v3["pairs"])[cols]
    p2 = pd.DataFrame(v2["pairs"])[cols]
    m = p3.merge(p2, on=["kind", "gnss", "reference", "ref_extra_mm"], suffixes=("_v3", "_v2"))
    out["A2_pairs_v3_vs_v2"] = m.round(1).to_dict(orient="records")
    print(m.round(1).to_string())
    steps = {}
    for ver in ("v3", "v2"):
        s = pd.read_csv(DATAPREP / "gnss_clean" / "preorigin" / ver / "steps.csv")
        s = s[(s.family == "wra") & s.station.isin(STATIONS) & (s.date >= "2019-12-01")
              & (s.date <= "2022-01-31")]
        steps[ver] = [f"{r.station} {r.date} {1000 * r.size_m:+.1f}mm z_own "
                      f"{r.z_own_series:.1f} {r.basis} flag{r.flag_after}"
                      for r in s.itertuples()]
        print(ver, steps[ver])
    out["A2_steps_2020_21"] = steps
    (ev / "post_test_anatomy.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
