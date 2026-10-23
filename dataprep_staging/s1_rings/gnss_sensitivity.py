"""After-the-fact sensitivity: GNSS-implied scales as a fourth line of group evidence.

Not part of the product (declaration S1a.rings_bridged_rules.v1: gnss_clean failed its gate,
and its A2 failure is GNSS-ring disagreement). gnss_clean/full/v1 WRA daily heights are read
as after_the_fact_analysis. For each ring site with a co-located WRA station (the S3 pairs):

* excess rate = GNSS subsidence rate - ring total shortening rate, in 2020-01..2021-12 and in
  2023-02..2025-03 (daily OLS against the survey-matched ring series);
* ring gap from GNSS = GNSS subsidence between the last pre and the first post survey (mean of
  the daily heights within +-3 days, flags 0, 1, 3) minus the mean excess x gap;
* b_gnss = (ring gap from GNSS - raw ring gap) / L; sd = [half the excess difference x gap]
  (+) [station high-frequency sd x sqrt(2/n_days)] (+) [the steps' correction sd], over L.

The assignment is then recomputed with this line added (s1lib.evaluate, extra_lines).

    tools/cpu_slot.sh python -m s1_rings.gnss_sensitivity
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings import s1const as K  # noqa: E402
from s1_rings import s1io  # noqa: E402
from s1_rings import s1lib as L  # noqa: E402
from s1_rings.build import real_evaluation  # noqa: E402

PAIRS = {"光復國小": "GFES", "宏崙國小": "HLES", "湖南國小": "HNES", "拯民國小": "JMES",
         "客厝國小": "KTES", "溪州國小": "SJES", "新生國小": "SSES", "秀潭國小": "STES",
         "土庫國中": "TKJS", "元長國小": "YCES"}
W_PRE = ("2020-01-01", "2021-12-31")
W_POST = ("2023-02-01", "2025-03-31")


def gnss_daily() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    prod = guard.open_product(K.GNSS_FULL_V1, "after_the_fact_analysis")
    d = prod.read_table("wra/daily.parquet", columns=["station", "date", "value", "sd", "flag"])
    st = prod.read_table("stations.csv")
    return d, st, list(prod.notes)


def at(h: pd.Series, t, tol=3) -> tuple[float, int]:
    w = h.loc[pd.Timestamp(t) - pd.Timedelta(days=tol):pd.Timestamp(t) + pd.Timedelta(days=tol)]
    w = w.dropna()
    return (float(w.mean()), len(w)) if len(w) else (float("nan"), 0)


def main() -> None:
    d, st, notes = gnss_daily()
    d = d[d["flag"].isin([0, 1, 3])]
    sig = dict(zip(st["station"], st["sigma_hf_m"], strict=False))
    piles, _ = s1io.piles(preorigin=False)
    pos, res = real_evaluation(piles)
    rows, extra = [], {}
    for site, sta in PAIRS.items():
        gs = res[site].gs
        x = d[d.station == sta]
        h = pd.Series(x["value"].to_numpy(float),
                      index=pd.DatetimeIndex(pd.to_datetime(x["date"])).astype("datetime64[ns]"))
        h = h.sort_index()
        S = (pos[site][gs["bot"]] - pos[site][gs["top"]]).dropna()
        ex = []
        for w in (W_PRE, W_POST):
            hw = h.loc[w[0]:w[1]].dropna()
            sw = S.loc[w[0]:w[1]]
            if len(hw) < 200 or len(sw) < 12:
                ex.append(np.nan)
                continue
            g_rate = -np.polyfit(L.years(hw.index), hw.to_numpy(), 1)[0]   # subsidence m/yr
            r_rate = -np.polyfit(L.years(sw.index), sw.to_numpy(), 1)[0]   # shortening m/yr
            ex.append(g_rate - r_rate)
        h0, n0 = at(h, gs["t_lp"])
        h1, n1 = at(h, gs["t_fp"])
        row = {"site": site, "station": sta, "excess_pre_mm_yr": 1000 * ex[0],
               "excess_post_mm_yr": 1000 * ex[1]}
        if not (np.all(np.isfinite(ex)) and np.isfinite(h0) and np.isfinite(h1)):
            row["ok"] = False
            rows.append(row)
            continue
        gnss_gap = -(h1 - h0)
        exm = 0.5 * (ex[0] + ex[1])
        c_from = gnss_gap - exm * gs["gap"]
        ring_raw = gs["S_lp"] - gs["S_fp"]
        b = (c_from - ring_raw) / gs["L"]
        s_hf = sig.get(sta, 0.005)
        sd = math.sqrt((abs(ex[0] - ex[1]) / 2 * gs["gap"]) ** 2
                       + s_hf ** 2 / max(n0, 1) + s_hf ** 2 / max(n1, 1)) / gs["L"]
        line = L.line_lr(b, sd)
        extra[site] = {"gnss": line}
        row.update({"ok": True, "gnss_gap_mm": 1000 * gnss_gap, "ring_gap_from_gnss_mm":
                    1000 * c_from, "ring_gap_raw_mm": 1000 * ring_raw, "b_gnss_ppm": 1e6 * b,
                    "sd_ppm": 1e6 * sd, "lr": line["lr"], "vote": line["vote"]})
        rows.append(row)
    _, res4 = real_evaluation(piles, extra_lines=extra)
    changes = []
    for s in K.SITES:
        a3, a4 = res[s].assign, res4[s].assign
        changes.append({"site": s, "assigned_main": a3["assigned"], "assigned_with_gnss":
                        a4["assigned"], "evidence_group_main": a3["evidence_group"],
                        "evidence_group_with_gnss": a4["evidence_group"],
                        "p_A_main": round(res[s].p_A, 4), "p_A_with_gnss": round(res4[s].p_A, 4),
                        "C_primary_main_cm": round(100 * res[s].primary["C"], 2),
                        "C_primary_with_gnss_cm": round(100 * res4[s].primary["C"], 2)})
    out = {"status": "after-the-fact sensitivity (gnss_clean gate failed); not used by the "
                     "product", "guard_notes": notes, "lines": rows, "assignment": changes}
    p = K.EVIDENCE / "gnss_sensitivity.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    members = {"S1a:gnss_implied_scale.2020-01_2025-03": {
        "none": "GNSS stations co-located with ring sites (no registered well, benchmark or cell)",
        "description": "GNSS-implied ring scales at 10 ring sites, after the fact"}}
    prereg3.record_look(list(members), "gnss_wra+rings:2020-01..2025-03", "S1a", "diagnose",
                        members=members,
                        note="S1a after-the-fact GNSS sensitivity (gnss_clean/full/v1, gate "
                             "failed): GNSS-implied scales and assignment with GNSS as a fourth "
                             "line; not used by rings_bridged")
    print(pd.DataFrame(rows).round(2).to_string())
    print(pd.DataFrame(changes).to_string())


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
