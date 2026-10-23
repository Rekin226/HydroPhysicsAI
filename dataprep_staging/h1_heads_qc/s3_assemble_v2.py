"""H1a v2 stage 3 runner (round 2): apply the v2 decisions (``decide_v2``, ``assemble_v2``),
build daily/monthly tables, events, BE table, O1 coherence (python -I, CPU slot). A copy of
the v1 runner ``s3_assemble.py`` with the v2 decision table, the v2 apply and carry functions
and the v2 event fields (level-persistence and segment statistics, pressure fall; the v2
jump catalogue ``jump_v2_<status>`` next to the stage-1 classes ``jump_<cls>``).

Usage: s3_assemble_v2.py WORKDIR BUILD_DIR [STAGE1_SUBDIR]
Reads WORKDIR/station_meta_clean.parquet, WORKDIR/<stage1>/*.pkl and
WORKDIR/decisions_v2_<stage1>.parquet. Writes into BUILD_DIR:
  heads_daily.parquet, heads_monthly.parquet, events.csv, barometric_efficiency.csv,
  o1_coherence.csv and be_out_of_sample_2023_2026.csv (validation inputs; not product
  outputs), assemble_log.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import assemble as A  # noqa: E402
from h1_heads_qc import assemble_v2 as A2  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402
from h1_heads_qc import screens as S  # noqa: E402
from h1_heads_qc import tides  # noqa: E402
from h1_heads_qc.pipeline import POST_MONTHS, PRE_MONTHS  # noqa: E402

RUN_BITS = [S.K_SENTINEL, S.K_SPIKE, S.K_FLAT, S.K_MAD, S.K_FAR, S.K_DROPFLAT, S.K_BOTTOM]
ORIGIN_DAY = pd.Timestamp(C.ORIGIN_SPLIT)


def ts(t):
    return pd.Timestamp(int(t), unit="s")


def detection_date(period: str, t1) -> str:
    """The time-slice date of an event row: the last day of data its detection relied on is
    pre-origin for pre-pass events (their start day is used) and post-origin for post-pass
    events (never earlier than 2023-01-01)."""
    d = ts(t1).normalize()
    if period == "post":
        d = max(d, ORIGIN_DAY)
    return d.strftime("%Y-%m-%d")


def coherence_rows(sid, res):
    out = []
    for part, tag in (("o1_pre", "pre"), ("o1_post", "post")):
        o = res.get(part)
        if not o:
            continue
        for ver in ("raw", "corr"):
            g = np.asarray(o[ver].get("gO1", []), float)
            out.append({"sid": sid, "period": tag, "version": ver, "n_months": int(len(g)),
                        "coh_O1": tides.coherence(g) if len(g) else np.nan,
                        "coh_M2": tides.coherence(np.asarray(o[ver].get("gM2", []), float))
                        if len(g) else np.nan,
                        "median_A_O1": float(np.median(o[ver]["A_O1"])) if len(g) else np.nan})
    return out


def main():
    work, build = Path(sys.argv[1]), Path(sys.argv[2])
    sub = sys.argv[3] if len(sys.argv) > 3 else "stage1"
    build.mkdir(parents=True, exist_ok=True)
    dec_all = pd.read_parquet(work / f"decisions_v2_{sub}.parquet")
    t00 = time.time()
    daily_rows, monthly_rows, events, be_rows, coh, log = [], [], [], [], [], {}
    oos_rows = []
    for p in sorted((work / sub).glob("*.pkl")):
        res = pickle.loads(Path(p).read_bytes())
        sid = res["sid"]
        n_pre = len(res["pre"]["t"]) if "pre" in res else 0
        n_post = len(res["post"]["t"]) if "post" in res else 0
        if n_pre + n_post == 0:
            log[sid] = {"skipped": "no samples in either record"}
            continue
        dec = dec_all[dec_all.sid == sid]
        dec_pre, dec_post = dec[dec.period == "pre"], dec[dec.period == "post"]
        wl = {}
        # ---------------------------------------------------------------- pre-origin
        d_pre = pd.DataFrame()
        if "pre" in res and len(res["pre"]["t"]):
            part = res["pre"]
            value, corr, sflag, kind, applied = A2.apply_part_v2(part, dec_pre, None)
            d_pre = A.daily_table(part["t"], part["v_raw"], part["L"], corr, value, sflag, kind)
            wl["applied_pre"] = applied
            for bit in RUN_BITS:
                for a, b, n in S.run_events(part["t"], kind, bit):
                    events.append({"sid": sid, "period": "pre", "kind": S.KIND_NAMES[bit],
                                   "t_start": str(ts(a)), "t_end": str(ts(b)), "n_samples": n,
                                   "flag": 8 if bit == S.K_SENTINEL else 9, "applied": "flagged",
                                   "date": detection_date("pre", a)})
        mad = A.mad_daily_diff(d_pre)
        mad_from = "pre_origin"
        if len(d_pre):
            bad = A.departure_screen(d_pre, mad)
            d_pre = A.apply_departures(d_pre, bad)
            for dday in d_pre.loc[bad, "date"]:
                events.append({"sid": sid, "period": "pre", "kind": "daily_departure",
                               "t_start": str(dday), "t_end": str(dday), "flag": 9,
                               "applied": "flagged", "date": dday.strftime("%Y-%m-%d")})
        m_pre = A.monthly_table(d_pre, PRE_MONTHS)
        # ---------------------------------------------------------------- post-origin
        d_post = pd.DataFrame()
        gap_days = None
        if n_pre and n_post:
            gap_days = (int(res["post"]["t"][0]) - int(res["pre"]["t"][-1])) / 86400.0
        carry = A2.carry_until_v2(A2.carry_from_pre_v2(dec_pre, gap_days), dec_post)
        if "post" in res and len(res["post"]["t"]):
            part = res["post"]
            value, corr, sflag, kind, applied = A2.apply_part_v2(part, dec_post, carry)
            d_post = A.daily_table(part["t"], part["v_raw"], part["L"], corr, value, sflag,
                                   kind)
            wl["applied_post"] = applied
            for bit in RUN_BITS:
                for a, b, n in S.run_events(part["t"], kind, bit):
                    events.append({"sid": sid, "period": "post", "kind": S.KIND_NAMES[bit],
                                   "t_start": str(ts(a)), "t_end": str(ts(b)), "n_samples": n,
                                   "flag": 8 if bit == S.K_SENTINEL else 9, "applied": "flagged",
                                   "date": detection_date("post", a)})
            if not np.isfinite(mad):
                mad = A.mad_daily_diff(d_post)
                mad_from = "post_origin_only_well"
            bad = A.departure_screen(d_post, mad, context=d_pre)
            d_post = A.apply_departures(d_post, bad)
            for dday in d_post.loc[bad, "date"]:
                events.append({"sid": sid, "period": "post", "kind": "daily_departure",
                               "t_start": str(dday), "t_end": str(dday), "flag": 9,
                               "applied": "flagged", "date": dday.strftime("%Y-%m-%d")})
            if carry:
                events.append({"sid": sid, "period": "post", "kind": "step_carried",
                               "t_start": str(ts(carry["from"])),
                               "t_end": str(ts(carry["until"])) if carry.get("until") else "",
                               "j1": carry["j"], "flag": carry["flag"],
                               "applied": "corrected" if carry["flag"] in (1, 3) else "flagged",
                               "backing": f"carried from the pre-origin step of "
                                          f"{ts(carry['t_step'])} ({carry.get('backing')}); "
                                          f"ends at {carry.get('until_reason')}",
                               "date": "2023-01-01"})
        m_post = A.monthly_table(d_post, POST_MONTHS)
        wl["carry"] = carry
        wl["mad_daily_diff"] = mad
        wl["mad_from"] = mad_from
        log[sid] = wl
        for d, tag in ((d_pre, "pre"), (d_post, "post")):
            if len(d):
                d = d.copy()
                d.insert(0, "sid", sid)
                d["period"] = tag
                daily_rows.append(d)
        for m, tag in ((m_pre, "pre"), (m_post, "post")):
            m = m.copy()
            m.insert(0, "sid", sid)
            m["period"] = tag
            monthly_rows.append(m)
        # ---------------------------------------------------------------- catalogue events
        for _, d in dec.iterrows():
            k = d["kind"]
            fl = d["flag"]
            fl = int(fl) if fl is not None and np.isfinite(fl) else np.nan
            if k == "jump":
                applied_kind = "catalogue"
                kname = f"jump_v2_{d['v2_status']}"
            elif k == "unpaired_offset_lt3m":
                applied_kind = "info_not_flagged"
                kname = k
            else:
                applied_kind = "corrected" if (k == "step" and fl in (1, 3)) else "flagged"
                kname = k
            if k in ("step", "excursion", "jump_edge"):
                t_s, t_e = d["t_lo"], d["t_hi"]
            else:
                t_s, t_e = d["t1"], d["t2"]
            events.append({"sid": sid, "period": d["period"], "kind": kname,
                           "t_start": str(ts(t_s)), "t_end": str(ts(t_e)),
                           "t_jump1": str(ts(d["t1"])),
                           "t_jump2": str(ts(d["t2"])) if k in A2.EPISODE_KINDS else "",
                           "j1": d["j1"], "j2": d.get("j2"), "side": d.get("side"),
                           "side_rule": d.get("side_rule"), "flag": fl,
                           "applied": applied_kind, "backing": d.get("backing"),
                           "meta_impossible": d.get("meta_impossible"),
                           "meta_dates": d.get("meta_dates"),
                           "coloc_covering": d.get("coloc_covering"),
                           "coloc_jumped": d.get("coloc_jumped"),
                           "neighbour_verdict": d.get("neighbour_verdict"),
                           "neighbour_why": d.get("neighbour_why"),
                           "neighbour_delta_m": d.get("neighbour_delta_m"),
                           "neighbour_sd_m": d.get("neighbour_sd_m"),
                           "neighbour_z": d.get("neighbour_z"),
                           "neighbours": d.get("neighbours"),
                           "spans_origin": d.get("spans_origin"),
                           "v2_status": d.get("v2_status"), "cls1": d.get("cls1"),
                           "cls2": d.get("cls2"),
                           "lc_delta": d.get("lc_delta"), "lc_ratio": d.get("lc_ratio"),
                           "lc_status": d.get("lc_status"), "lc_delta2": d.get("lc_delta2"),
                           "lc_ratio2": d.get("lc_ratio2"), "lc_status2": d.get("lc_status2"),
                           "seg_delta": d.get("seg_delta"), "seg_ratio": d.get("seg_ratio"),
                           "seg_status": d.get("seg_status"),
                           "pressure_fall_hpa": d.get("pressure_fall_hpa"),
                           "date": detection_date(d["period"], d["t1"])})
        for part in ("pre_cat", "post_cat"):
            cat = res.get(part)
            if not cat:
                continue
            for J in cat["jumps"]:
                events.append({"sid": sid, "period": J["period"], "kind": f"jump_{J['cls']}",
                               "t_start": str(ts(J["t"])), "t_end": str(ts(J["t"])),
                               "j1": J["j"], "flag": np.nan, "applied": "catalogue",
                               "date": detection_date(J["period"], J["t"])})
        # ---------------------------------------------------------------- BE table
        be = res.get("be", {})
        oos = res.get("be_oos", {}) or {}
        segs = be.get("segments") or [{}]
        # slice date: every parameter of the row was fitted on 2012-2022 data, except the MAD
        # reference of a well with no pre-origin record (post-origin rows only)
        sdate = "2022-12-31" if res.get("ref_window") == "pre_origin" else "2023-01-01"
        for x in segs:
            be_rows.append({
                "sid": sid, "era5_cell": res.get("cell"), "epoch": x.get("epoch", 0),
                "n_epochs": x.get("n_epochs", 1),
                "epoch_start": str(ts(x["start"])) if x.get("start") is not None else "",
                "epoch_end": str(ts(x["end"])) if x.get("end") is not None else "",
                "applies_to_post_origin": bool(x is segs[-1] and x.get("a") is not None),
                "design": x.get("design"), "status": x.get("status", be.get("status")),
                "accepted": bool(x.get("accepted", False)), "A_K": x.get("A_K"),
                "se_A_K": x.get("se_A_K"), "se_A_K_bootstrap": x.get("se_A_K_bootstrap"),
                "BE_long": x.get("BE_long"),
                "BE_lag0": x.get("BE_lag0"), "BE_lag6": x.get("BE_lag6"),
                "BE_lag24": x.get("BE_lag24"), "partial_r2": x.get("partial_r2"),
                "resid_sd_m": x.get("resid_sd_m"), "n_pairs": x.get("n_pairs"),
                "n_blocks": x.get("n_blocks"), "sensor_type_hint": x.get("sensor_type_hint"),
                "p_ref_hpa": res.get("p_ref_hpa"), "fit_window": "2012-01..2022-12",
                "impulse_response_a_0_48h": json.dumps([round(v, 6) for v in x["a"]])
                if x.get("a") is not None else "",
                "med_ref": res.get("med_ref"), "mad_ref": res.get("mad_ref"),
                "ref_window": res.get("ref_window"), "date": sdate})
        oos_rows.append({"sid": sid, "oos_ratio_robust": oos.get("ratio_robust"),
                         "oos_ratio_plain": oos.get("ratio_plain"),
                         "oos_n_pairs": oos.get("n_pairs")})
        coh += coherence_rows(sid, res)
        print(f"{sid} {time.time() - t00:.0f}s", flush=True)

    D = pd.concat(daily_rows, ignore_index=True)
    D["kinds_names"] = D["kinds"].map(A2.kinds_to_names_v2)
    D = D[["sid", "date", "period", "value", "flag", "day_valid", "n_samples", "n_hours",
           "value_raw", "baro_m", "corr_m", "n_suspect", "n_sentinel", "kinds", "kinds_names"]]
    D["flag"] = D["flag"].astype("uint8")
    D = D.sort_values(["sid", "date"]).reset_index(drop=True)
    M = pd.concat(monthly_rows, ignore_index=True)
    M = M[["sid", "date", "month", "period", "value", "flag", "c20", "n_clean_days",
           "sigma_obs", "n_valid_days_any", "n_days_any", "n_days_flag9", "n_days_flag3",
           "days_in_month", "value_raw", "baro_m", "corr_m"]]
    M["flag"] = M["flag"].astype("uint8")
    M = M.sort_values(["sid", "date"]).reset_index(drop=True)
    E = pd.DataFrame(events)
    E["meta_dates"] = E["meta_dates"].map(lambda x: x if isinstance(x, str) else
                                          (",".join(x) if isinstance(x, (list, np.ndarray))
                                           else ""))
    E = E.sort_values(["sid", "t_start", "kind"]).reset_index(drop=True)
    B = pd.DataFrame(be_rows)
    Cc = pd.DataFrame(coh)
    D.to_parquet(build / "heads_daily.parquet", index=False)
    M.to_parquet(build / "heads_monthly.parquet", index=False)
    E.to_csv(build / "events.csv", index=False)
    B.to_csv(build / "barometric_efficiency.csv", index=False)
    Cc.to_csv(build / "o1_coherence.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(build / "be_out_of_sample_2023_2026.csv", index=False)
    Path(build / "assemble_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(f"daily {len(D)} monthly {len(M)} events {len(E)} wells {len(B)} "
          f"in {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
