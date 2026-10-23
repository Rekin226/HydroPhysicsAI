"""WP S1b: the registered checks (c) and (d) of the ring bridge, and their report-only variants.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_run.py registered
    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_run.py after

``registered``: refuses to run unless the pre-registration S1b.ring_bridge_checks.v1 is intact
and predates every look at its test data (``harness.prereg3.assert_can_score``, which logs the
scoring looks); computes the primary configuration; writes
``S1b_evidence/registered.json`` and ``registered_rows.csv``.

``after``: logs an after-the-fact look, then computes the report-only variants listed in the
registration; writes ``S1b_evidence/after_the_fact.json``. Nothing here changes a rule, a
threshold or the registered result, and nothing feeds back into S1a.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_core as C  # noqa: E402
from s1_rings.s1b import s1b_io as IO  # noqa: E402

PRIMARY = {"gnss_flags": tuple(K.GNSS_FLAGS), "steps_mode": "product", "common_mode": None,
           "seasonal": True, "insar_source": "pixels", "segment_intercepts": True,
           "tol_sigma": K.TOL_SIGMA, "exclude_stations": ()}


def line(b_ppm: float, sd_ppm: float) -> dict:
    """S1a's evidence line (s1lib.line_lr, imported read-only): likelihood ratio A:B of a
    scale estimate against the plan's reference groups, and its vote. Report only."""
    from s1_rings import s1lib
    if not (np.isfinite(b_ppm) and np.isfinite(sd_ppm)):
        return {"lr": None, "vote": None}
    r = s1lib.line_lr(b_ppm * 1e-6, sd_ppm * 1e-6)
    return {"lr": float(r["lr"]), "vote": r.get("vote") or ""}


def _row_base(site: str, r: dict) -> dict:
    return {"site": site, "site_en": K.SITE_EN[site], "assigned": r["assigned"],
            "primary_tag": r["primary"], "p_A": r["p_A"], "L_m": r["L_mm"] / 1000.0,
            "gap_yr": r["gap_yr"], "t_lp": str(pd.Timestamp(r["t_lp"]).date()),
            "t_fp": str(pd.Timestamp(r["t_fp"]).date()), "sd_s_mm": r["sd_s_mm"],
            "C_primary_mm": r["C"]["primary"], "C_A_mm": r["C"]["A"], "C_B_mm": r["C"]["B"],
            "C_raw_mm": r["C"]["raw"]}


def _finish(row: dict, res: dict, var_end: float, r: dict, cfg: dict, n_br: int = 0,
            br_sd: float = 0.0, var_steps: float = 0.0) -> dict:
    row.update({k: res.get(k) for k in ("pre_window", "post_window", "n_inst_pre",
                                        "n_inst_post", "n_ring_pre", "n_ring_post",
                                        "n_end_lp", "n_end_fp", "n_bridges", "n_seg_post",
                                        "rate_inst_pre", "rate_ring_pre", "rate_inst_post",
                                        "rate_ring_post")})
    if not res["ok"]:
        row.update({"evaluable": False, "reason": res["reason"]})
        return row
    ex = res["excess"]
    chk = C.site_check(res["G_mm"], ex, var_end, r["sd_s_mm"], r["L_mm"], r["C"],
                       k=cfg["tol_sigma"], n_bridges=n_br, bridge_sd=br_sd,
                       var_steps=var_steps)
    row.update({"evaluable": True, "reason": "", "excess_rate_pre": ex["excess_rate_pre"],
                "excess_rate_post": ex["excess_rate_post"], "seasonal_mm": ex["seasonal_mm"],
                "var_end_inst": var_end, **{k: v for k, v in chk.items()}})
    ln = line(chk["b_mid_ppm"], chk["b_sd_ppm"])
    row["line_lr_A_B"], row["line_vote"] = ln["lr"], ln["vote"]
    return row


def gnss_row(site: str, pair: dict, rings: dict, g: dict, cfg: dict) -> dict:
    r = rings[site]
    row = {**_row_base(site, r), "family": pair["family"], "station": pair["station"],
           "d_km": pair["d_km"]}
    t, d, var_steps = IO.gnss_series(g, pair["family"], pair["station"],
                                     flags=cfg["gnss_flags"], steps_mode=cfg["steps_mode"],
                                     gap=(r["t_lp"], r["t_fp"]),
                                     common_mode=cfg["common_mode"])
    res = C.instrument_vs_ring(
        ring_t=r["t"], ring_c=r["c"], inst_t=t, inst_d=d, t_lp=r["t_lp"], t_fp=r["t_fp"],
        pre_win=K.PRE_WIN, post_win=K.POST_WIN, min_inst=K.MIN_GNSS_DAYS,
        min_ring=K.MIN_RING_SURVEYS, min_span=K.MIN_SPAN_DAYS, end_half_d=K.GNSS_END_HALF_D,
        end_min=K.GNSS_END_MIN, end_half_fallback=K.GNSS_END_HALF_D_FALLBACK,
        seasonal=cfg["seasonal"], segment_intercepts=False)
    var_end = (IO.gnss_end_var(g["stations"], pair["family"], pair["station"],
                               res["n_end_lp"], res["n_end_fp"]) if res["ok"] else float("nan"))
    return _finish(row, res, var_end, r, cfg, var_steps=var_steps)


def insar_site(site: str, rings: dict, ins: dict, xy: dict, cfg: dict) -> dict:
    r = rings[site]
    x, y = xy[site]
    cell = IO.twin_cell(x, y)
    row = {**_row_base(site, r), "family": "insar", "station": f"069A:{cfg['insar_source']}",
           "cell": cell}
    if cfg["insar_source"] == "pixels":
        px = IO.insar_pixel_series(ins, x, y)
        t, d, seg = ins["dates"], px["down_mm"], ins["network_segment"]
        row.update({"pixel_dist_m": px["dist_m"], "n_pix_ok": px["n_pix_ok"]})
    else:
        cs = IO.insar_cell_series(ins, cell)
        t, d, seg = ins["cell_dates"], cs["down_mm"], cs.get("segment")
        if seg is None:
            seg = np.zeros(len(t), int)
    # (d1) gap change from the endpoints alone (no fit)
    e0 = C.endpoint_mean(t, d, r["t_lp"], K.INSAR_END_HALF_D, K.INSAR_END_MIN, seg)
    e1 = C.endpoint_mean(t, d, r["t_fp"], K.INSAR_END_HALF_D, K.INSAR_END_MIN, seg)
    row["I_gap_mm"] = e1["mean"] - e0["mean"]
    row["I_bridges"] = (int(e1["seg"]) - int(e0["seg"])) if (e0["seg"] is not None and
                                                           e1["seg"] is not None) else None
    res = C.instrument_vs_ring(
        ring_t=r["t"], ring_c=r["c"], inst_t=t, inst_d=d, inst_seg=seg, t_lp=r["t_lp"],
        t_fp=r["t_fp"], pre_win=K.PRE_WIN, post_win=K.POST_WIN, min_inst=K.MIN_INSAR_EPOCHS,
        min_ring=K.MIN_RING_SURVEYS, min_span=K.MIN_SPAN_DAYS, end_half_d=K.INSAR_END_HALF_D,
        end_min=K.INSAR_END_MIN, seasonal=cfg["seasonal"],
        segment_intercepts=cfg["segment_intercepts"])
    var_end = (K.INSAR_EPOCH_SD_MM ** 2 * (1.0 / res["n_end_lp"] + 1.0 / res["n_end_fp"])
               if res["ok"] else float("nan"))
    return _finish(row, res, var_end, r, cfg, n_br=res.get("n_bridges", 0) or 0,
                   br_sd=K.INSAR_BRIDGE_SD_MM)


def d1(rows: list[dict], *, drop: tuple = (), use_change: bool = False) -> dict:
    """(d1): Pearson r across sites of the InSAR gap-mean subsidence rate with the bridged and
    the raw gap-mean compaction rates (or the changes with ``use_change``)."""
    rr = [r for r in rows if np.isfinite(r.get("I_gap_mm", np.nan)) and r["site"] not in drop]
    f = (lambda r: 1.0) if use_change else (lambda r: 1.0 / r["gap_yr"])
    x = [r["I_gap_mm"] * f(r) for r in rr]
    yb = [r["C_primary_mm"] * f(r) for r in rr]
    yr = [r["C_raw_mm"] * f(r) for r in rr]
    out = C.corr_check(x, yb, yr, K.CORR_ALPHA)
    out["sites"] = [r["site"] for r in rr]
    return out


def compute(cfg: dict, rings: dict, xy: dict, g: dict, ins: dict) -> dict:
    prs = IO.pairs(g["stations"], xy)
    ex = set(cfg.get("exclude_stations") or ())
    prs = {k: [p for p in v if p["station"] not in ex] for k, v in prs.items()}
    rows = {"c_wra": [gnss_row(p["site"], p, rings, g, cfg) for p in prs["wra_colocated"]],
            "c_ngl": [gnss_row(p["site"], p, rings, g, cfg) for p in prs["ngl_near"]],
            "c_neighbour": [gnss_row(p["site"], p, rings, g, cfg) for p in prs["neighbour"]],
            "d_insar": [insar_site(s, rings, ins, xy, cfg) for s in K.SITES]}
    v = {k: C.family_verdict(rows[k], K.FAMILY_MIN_PASS_FRAC, K.FAMILY_MIN_SITES_GATE)
         for k in ("c_wra", "c_ngl", "c_neighbour", "d_insar")}
    corr = d1(rows["d_insar"])
    gate_parts = {"c_wra": v["c_wra"], "c_ngl": v["c_ngl"], "d2_insar": v["d_insar"]}
    in_gate = {k: x["pass"] for k, x in gate_parts.items() if x["in_gate"]}
    in_gate["d1_corr"] = corr["pass"]
    return {"config": {k: (list(x) if isinstance(x, tuple) else x) for k, x in cfg.items()},
            "pairs": prs, "rows": rows, "verdicts": v, "d1": corr,
            "gate_members": in_gate, "gate_pass": bool(all(in_gate.values())),
            "not_in_gate": [k for k, x in gate_parts.items() if not x["in_gate"]]}


def load_all() -> tuple[dict, dict, dict, dict, list]:
    rings, n1 = IO.rings()
    g = IO.gnss()
    ins = IO.insar()
    return rings, IO.site_xy(), g, ins, n1 + g["notes"] + ins["notes"]


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def dump(obj, path: Path) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path.write_text(json.dumps(clean(obj), indent=1, ensure_ascii=False, default=_js))


def rows_frame(res: dict) -> pd.DataFrame:
    out = []
    for test, rows in res["rows"].items():
        for r in rows:
            out.append({"test": test, **{k: v for k, v in r.items()
                                         if not isinstance(v, (list, dict))}})
    return pd.DataFrame(out)


def registered() -> dict:
    from harness import prereg3
    ok = prereg3.assert_can_score(K.PREREG_ID, K.WP)
    rings, xy, g, ins, notes = load_all()
    res = compute(PRIMARY, rings, xy, g, ins)
    res.update({"prereg": K.PREREG_ID, "prereg_sha256": ok["sha256"],
                "post_registration_looks": ok["post_registration_looks"],
                "unverifiable_prior_looks": ok["unverifiable_prior_looks"],
                "guard_notes": notes,
                "raw_check_max_abs_mm": max(abs(r["raw_check_mm"]) for r in rings.values())})
    K.EVIDENCE.mkdir(parents=True, exist_ok=True)
    dump(res, K.EVIDENCE / "registered.json")
    rows_frame(res).to_csv(K.EVIDENCE / "registered_rows.csv", index=False)
    return res


VARIANTS = {
    "gnss_flags_0_1": {"gnss_flags": (0, 1)},
    "gnss_catalogue_steps_only_in_gap": {"steps_mode": "catalogue_in_gap"},
    "gnss_ngl_offfan_removed": {"common_mode": "ngl_offfan"},
    "no_seasonal": {"seasonal": False},
    "insar_cells": {"insar_source": "cells"},
    "insar_no_segment_intercepts": {"segment_intercepts": False},
    "tolerance_1sigma": {"tol_sigma": 1.0},
    "plan_flagged_SJES_excluded": {"exclude_stations": ("SJES",)},
}


def after() -> dict:
    from harness import prereg3
    m = {"S1b:after_the_fact.variants_2026-10-14": {
        "none": "ring sites, their GNSS stations and InSAR pixels (no registered member)",
        "description": "report-only variants of the registered S1b checks, after scoring"}}
    prereg3.record_look(sorted(m), "rings+gnss+insar:2020-01..2025-12", K.WP,
                        "after_the_fact", members=m, prereg_id=K.PREREG_ID,
                        note="report-only variants registered in S1b.ring_bridge_checks.v1 "
                             "(GNSS flags 0-1, catalogue-only in-gap steps, ngl_offfan "
                             "removed, no seasonal term, InSAR cells, no segment intercepts, "
                             "1 sigma, SJES excluded), computed after the registered run; "
                             "never a gate")
    rings, xy, g, ins, notes = load_all()
    out = {"variants": {}}
    for name, over in VARIANTS.items():
        cfg = {**PRIMARY, **over}
        res = compute(cfg, rings, xy, g, ins)
        out["variants"][name] = {"config": res["config"], "verdicts": res["verdicts"],
                                 "d1": res["d1"], "gate_pass": res["gate_pass"],
                                 "rows": res["rows"]}
    base = compute(PRIMARY, rings, xy, g, ins)
    out["d1_variants"] = {
        "changes_not_rates": d1(base["rows"]["d_insar"], use_change=True),
        "without_beichen": d1(base["rows"]["d_insar"], drop=("北辰國小",)),
        "pilot_9_sites": d1(base["rows"]["d_insar"], drop=tuple(
            s for s in K.SITES if s not in ("元長國小", "客厝國小", "內寮派駐站", "土庫國中",
                                            "秀潭國小", "宏崙國小", "光復國小", "拯民國小",
                                            "嘉興國小"))),
    }
    xs = [r for r in base["rows"]["d_insar"] if np.isfinite(r.get("I_gap_mm", np.nan))]
    from scipy import stats
    out["d1_spearman"] = {
        "rho_bridged": float(stats.spearmanr([r["I_gap_mm"] / r["gap_yr"] for r in xs],
                                             [r["C_primary_mm"] / r["gap_yr"] for r in xs])[0]),
        "rho_raw": float(stats.spearmanr([r["I_gap_mm"] / r["gap_yr"] for r in xs],
                                         [r["C_raw_mm"] / r["gap_yr"] for r in xs])[0]),
        "n": len(xs)}
    out["guard_notes"] = notes
    dump(out, K.EVIDENCE / "after_the_fact.json")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else ""
    if what == "registered":
        res = registered()
        print(json.dumps({"gate_pass": res["gate_pass"], "gate_members": res["gate_members"],
                          "verdicts": res["verdicts"], "d1": res["d1"]}, indent=1,
                         ensure_ascii=False, default=_js))
    elif what == "after":
        out = after()
        print(json.dumps({k: {"verdicts": v["verdicts"], "d1": v["d1"]}
                          for k, v in out["variants"].items()}, indent=1, ensure_ascii=False,
                         default=_js)[:6000])
    else:
        raise SystemExit("usage: s1b_run.py registered|after")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
