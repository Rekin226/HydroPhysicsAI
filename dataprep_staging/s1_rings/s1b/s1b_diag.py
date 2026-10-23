"""WP S1b: after-the-fact diagnostics of the registered checks (never a gate, never fed back).

    python -I dataprep_staging/s1_rings/s1b/s1b_diag.py

Logs a ``diagnose`` look first, then:

1. SJES (Xizhou, and Qiaoyi's neighbour): the GNSS gap change is 125.8 mm against 21.4 mm from
   InSAR and about 11-30 mm implied by the rings. Monthly means of the raw and corrected daily
   heights around the gap, the product's step corrections, and the gap change with and
   without the 2022-03-29 correction.
2. InSAR post-gap rates: at the 10 co-located sites, the InSAR minus GNSS subsidence rate in
   the pre- and post-gap windows, with and without segment intercepts (from the registered
   rows and the registered no-intercept variant), and the post-gap segment layout.
3. Instrument-implied scales b_mid +- b_sd per site and family (registered rows) next to the
   product's group scales, and the per-site membership evidence table.

Writes ``S1b_evidence/diagnostics.json``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_io as IO  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402


def log_look() -> None:
    from harness import prereg3
    m = {"S1b:diagnose.sjes_and_insar_rates_2026-10-14": {
        "none": "SJES daily heights 2021-06..2023-12, the registered S1b rows and variant rows "
                "(no registered member)",
        "description": "after-the-fact diagnostics of S1b.ring_bridge_checks.v1"}}
    prereg3.record_look(sorted(m), "rings+gnss+insar:2020-01..2025-05", K.WP, "diagnose",
                        members=m, prereg_id=K.PREREG_ID,
                        note="after the registered run and the registered variants: SJES gap "
                             "change (raw vs corrected monthly means around the gap), InSAR "
                             "minus GNSS rates at co-located sites in both windows, "
                             "instrument-implied scales per site; report only")


def sjes(g: dict, rings: dict) -> dict:
    d = g["daily"]
    x = d[(d.family == "wra") & (d.station == "SJES")].copy()
    x = x[(x.date >= "2021-06-01") & (x.date <= "2023-12-31")]
    x["month"] = x["date"].dt.to_period("M").astype(str)
    mon = x.groupby("month").agg(n=("value", "size"), raw_mm=("value_raw", "mean"),
                                 corr_mm=("value", "mean"), correction_mm=("correction_m",
                                                                           "mean"),
                                 n_flag9=("flag", lambda f: int((f == 9).sum())))
    for c in ("raw_mm", "corr_mm", "correction_mm"):
        mon[c] = 1000.0 * mon[c]
    base = mon.loc["2021-11", ["raw_mm", "corr_mm"]]
    mon["raw_rel_mm"] = mon["raw_mm"] - base["raw_mm"]
    mon["corr_rel_mm"] = mon["corr_mm"] - base["corr_mm"]
    st = g["steps"]
    st = st[(st.family == "wra") & (st.station == "SJES")]
    r = rings["溪州國小"]
    out = {"t_lp": str(r["t_lp"].date()), "t_fp": str(r["t_fp"].date()),
           "monthly": mon.round(2).reset_index().to_dict(orient="records"),
           "steps": st[["date", "size_m", "se_m", "evidence", "basis", "flag_after"]]
           .to_dict(orient="records")}
    # gap change (subsidence, mm) from +-7 d endpoint means, raw / corrected / corrected with
    # the 2022-03-29 correction undone
    full = d[(d.family == "wra") & (d.station == "SJES")]
    res = {}
    for name, col in (("raw", "value_raw"), ("corrected", "value")):
        v = full[col].to_numpy(float)
        t = pd.DatetimeIndex(full["date"])
        e = []
        for c in (r["t_lp"], r["t_fp"]):
            sel = np.abs((t - c).days.to_numpy()) <= 7
            e.append(float(np.nanmean(v[sel])) if sel.any() else float("nan"))
        res[name] = -1000.0 * (e[1] - e[0])
    out["gap_change_mm"] = res
    out["ring_C"] = {k: v for k, v in r["C"].items()}
    return out


def rates(reg: pd.DataFrame, var_rows: list[dict]) -> list[dict]:
    g = reg[reg.test == "c_wra"].set_index("site")
    i = reg[reg.test == "d_insar"].set_index("site")
    nv = {r["site"]: r for r in var_rows if r.get("evaluable")}
    out = []
    for s in g.index:
        out.append({"site": K.SITE_EN[s], "station": g.loc[s, "station"],
                    "gnss_pre": g.loc[s, "rate_inst_pre"], "gnss_post": g.loc[s, "rate_inst_post"],
                    "insar_pre": i.loc[s, "rate_inst_pre"],
                    "insar_post_seg": i.loc[s, "rate_inst_post"],
                    "insar_post_noseg": nv[s]["rate_inst_post"] if s in nv else None,
                    "ring_pre": g.loc[s, "rate_ring_pre"], "ring_post": g.loc[s, "rate_ring_post"]})
    df = pd.DataFrame(out)
    df["insar_minus_gnss_pre"] = df.insar_pre - df.gnss_pre
    df["insar_minus_gnss_post_seg"] = df.insar_post_seg - df.gnss_post
    df["insar_minus_gnss_post_noseg"] = df.insar_post_noseg - df.gnss_post
    return df.round(2).to_dict(orient="records")


def main() -> int:
    log_look()
    rings, _ = IO.rings()
    g = IO.gnss()
    ins = IO.insar()
    reg = pd.read_csv(K.EVIDENCE / "registered_rows.csv")
    aft = json.loads((K.EVIDENCE / "after_the_fact.json").read_text())
    out = {"sjes": sjes(g, rings),
           "rates_colocated_mm_yr": rates(reg, aft["variants"]["insar_no_segment_intercepts"]
                                          ["rows"]["d_insar"]),
           "insar_network_segments": {"dates_first_last_per_segment": {
               int(s): [str(ins["dates"][ins["network_segment"] == s].min().date()),
                        str(ins["dates"][ins["network_segment"] == s].max().date())]
               for s in np.unique(ins["network_segment"])}}}
    rr = reg[reg.evaluable.astype(bool)]
    out["scales"] = rr[["test", "site_en", "station", "p_A", "primary_tag", "b_mid_ppm",
                        "b_sd_ppm", "membership_evidence", "line_lr_A_B", "line_vote",
                        "contains_primary", "contains_raw"]].round(3).to_dict(orient="records")
    dump(out, K.EVIDENCE / "diagnostics.json")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out["sjes"]["monthly"]).to_string())
    print(out["sjes"]["gap_change_mm"], out["sjes"]["steps"])
    print(pd.DataFrame(out["rates_colocated_mm_yr"]).to_string())
    print(out["insar_network_segments"])
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
