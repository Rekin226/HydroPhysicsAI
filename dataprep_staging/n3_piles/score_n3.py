"""Score N3.piles_prospective.v1 once, after the window closes (not before 2028-01-15).

Order of a real run: (1) the pile product (deep_benchmarks_printed v3, future months only)
and the window heads product must grant prospective_scorer - MANIFEST metadata only,
otherwise not evaluable, nothing read, no look; (2) order: N2's score look and N2C's
step-(1) score look must be logged (and, for the 溪州 arm, N2C's step-(2) report must exist),
otherwise REFUSED, nothing read, no look; (3) prereg3.assert_can_score per test (logs the
look); (4) only then pile and head rows are read. Every number comes from n3lib (hashed in
the registration); the roles (gating / report-only) are read from the registration itself.

Usage (repository root, the lead):
    python -I dataprep_staging/n3_piles/score_n3.py score --horizon 12
        --piles-product results/dataprep/deep_benchmarks_printed/v3
        --heads-product <window heads product> [--heads-table heads_monthly.parquet]
        [--stations <station metadata parquet>] --n2c-report <N2C report.json>
        --out results/dataprep/reports/N3/score_12m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n3lib as L  # noqa: E402
import well_columns as WC  # noqa: E402

SCORER_VERSION = "1.0.0"
NOT_SCORABLE = "not_scorable"
PILE_TABLE = "integrated_monthly.parquet"
WELL_TABLE = "wells_monthly.parquet"
WORDING = {
    "conversion_supported": "compaction per metre of head change at the 2 Yunlin deep-pile "
                            "cells within x1.5 of the model's (rho {rho:.2f}, 90 % "
                            "{lo:.2f}-{hi:.2f}; N3, prospective)",
    "direction_supported_size_not": "the sign of the compaction-head link is supported at the "
                                    "2 Yunlin pile cells; its size is not (rho {rho:.2f}, 90 % "
                                    "{lo:.2f}-{hi:.2f}); no correction factor follows",
    "not_supported": "compaction per metre of head change at the 2 Yunlin pile cells: not "
                     "supported (rho {rho:.2f}, 90 % {lo:.2f}-{hi:.2f})",
    "not_informative": "undecided ({why})",
    "level_supported": "status-quo compaction at the 2 Yunlin pile cells beat zero and the "
                       "pre-origin leveling-cell rate (N3, prospective)",
    "level_not_supported": "status-quo compaction at the 2 Yunlin pile cells did not beat "
                           "both zero and the pre-origin leveling-cell rate (N3, prospective)",
    "not_evaluable": "undecided ({why})",
}


class ScoringRefused(RuntimeError):
    """The scorer refuses to run (date, order or provenance)."""


class EngineeringFailure(ScoringRefused):
    """An input is inconsistent (no verdict; fix, log and re-run)."""


# ----------------------------------------------------------------------------- checks
def product_grants(product_dir, use: str = "prospective_scorer") -> tuple[bool, str]:
    """MANIFEST metadata only (no table read)."""
    from common import guard
    from common.manifest import load_manifest

    try:
        m, _ = load_manifest(product_dir)
        guard.check_use(m, use)
    except Exception as exc:  # noqa: BLE001 - any refusal means 'not granted'
        return False, str(exc)
    return True, f"MANIFEST grants {use}"


def _score_looks(prereg_dir) -> set:
    from harness import prereg3

    return {(lk.get("prereg_id"), lk.get("test_id"))
            for lk in prereg3.read_chained(Path(prereg_dir) / "looks.jsonl")
            if lk.get("purpose") == "score"}


def order_status(prereg_dir, n2c_report, need_step2: bool) -> dict:
    """N2's score look and N2C's step-(1) look (and for the 溪州 arm N2C's step-(2) report,
    a JSON naming N2C, or a lead-written record with verdict 'not_scorable')."""
    have = _score_looks(prereg_dir)
    st = {"n2_score_look": (L.N2_ID, L.N2_TEST) in have,
          "n2c_step1_look": (L.N2C_ID, L.N2C_TEST) in have, "n2c_step2_report": False}
    if n2c_report and Path(n2c_report).is_file():
        try:
            r = json.loads(Path(n2c_report).read_text(encoding="utf-8"))
            st["n2c_step2_report"] = isinstance(r, dict) and r.get("prereg_id") == L.N2C_ID
            st["n2c_not_scorable"] = st["n2c_step2_report"] and r.get("verdict") == NOT_SCORABLE
        except (OSError, ValueError):
            pass
    ok = st["n2_score_look"] and (st["n2c_step1_look"] or st.get("n2c_not_scorable", False))
    st["ok_piles"] = bool(ok)
    st["ok_xizhou"] = bool(ok and st["n2c_step2_report"])
    return st


# ----------------------------------------------------------------------------- data shaping
def pile_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Verified (flag 0) rows -> pile, depth_m, month, inc_cm, year_cm."""
    r = rows[rows["flag"] == 0].copy()
    r["inc_cm"] = pd.to_numeric(r["value"]) / 10.0
    yv = r["year_value"] if "year_value" in r.columns else pd.Series(np.nan, index=r.index)
    r["year_cm"] = pd.to_numeric(yv, errors="coerce") / 10.0
    return r[["pile", "depth_m", "month", "inc_cm", "year_cm"]]


def choose_sensor(pf: pd.DataFrame, key: str, span) -> tuple[int | None, pd.Series]:
    """The declared sensor of a pile and its verified monthly increments (cm) in ``span``:
    the primary depth; 土庫's 300 m only if 400 m has < MIN_MONTHS months and 300 m >= it."""
    spec = L.PILES[key]
    need = L.MIN_MONTHS if span == L.WINDOW else L.MIN_MONTHS24

    def inc(depth):
        s = pf[(pf["pile"] == spec["pile"]) & (pf["depth_m"] == depth)
               & (pf["month"] >= span[0]) & (pf["month"] <= span[1])]
        return s.drop_duplicates("month").set_index("month")["inc_cm"].dropna()
    s = inc(spec["depth_m"])
    if len(s) < need and spec.get("fallback_depth_m"):
        f = inc(spec["fallback_depth_m"])
        if len(f) >= need:
            return spec["fallback_depth_m"], f
    return spec["depth_m"], s


def window_total(pf: pd.DataFrame, key: str, depth: int, span) -> dict:
    """Observed window compression (cm): printed trailing-12 value(s) at the declared
    sensor; else the summed verified months (all needed); else None."""
    pile = L.PILES[key]["pile"]
    g = pf[(pf["pile"] == pile) & (pf["depth_m"] == depth)].drop_duplicates("month")
    g = g.set_index("month")
    ends = [L.WINDOW[1]] + ([L.WINDOW24[1]] if span == L.WINDOW24 else [])
    ys = [g["year_cm"].get(m, np.nan) for m in ends]
    ms = L.months(*span)
    inc = [g["inc_cm"].get(m, np.nan) for m in ms]
    out = {"trailing12": float(np.sum(ys)) if np.isfinite(ys).all() else None,
           "summed": float(np.sum(inc)) if np.isfinite(inc).all() else None}
    out["value"] = out["trailing12"] if out["trailing12"] is not None else out["summed"]
    out["construction"] = ("trailing12" if out["trailing12"] is not None else
                           "summed" if out["summed"] is not None else None)
    return out


def heads_series(hf: pd.DataFrame, ids) -> dict[str, pd.Series]:
    """{sid: Series by 'YYYY-MM'} of the window heads (flags 0-1 already applied)."""
    hf = hf.copy()
    if "month" not in hf.columns:
        if "date" not in hf.columns:
            raise EngineeringFailure("heads table has neither 'month' nor 'date'")
        hf["month"] = hf["date"].astype(str).str[:7]
    hf["month"] = hf["month"].astype(str).str[:7]
    hf["sid"] = hf["sid"].astype(str)
    out = {}
    for sid in ids:
        g = hf[hf["sid"] == str(sid)]
        if len(g):
            out[str(sid)] = g.groupby("month")["value"].mean().astype(float)
    return out


# ----------------------------------------------------------------------------- the tests
def conversion(ms: dict, mapping: dict, pf: pd.DataFrame, heads: dict, span, role: str,
               fallback_changes: dict | None = None) -> dict:
    g_key = "g_mod_12" if span == L.WINDOW else "g_mod_24"
    g_mod = float(ms[g_key]["g_mod"])
    need = L.MIN_MONTHS if span == L.WINDOW else L.MIN_MONTHS24
    ys, xs, x1, x2, mi, gs, per = [], [], [], [], [], [], {}
    for key in L.PRIMARY:
        nest = mapping["piles"][key]["nest"]
        lay = {w["sid"]: w["layer"] for w in nest}
        ms_list = L.months(L.prev_month(span[0], 2), span[1])
        hh = {s: heads[s] for s in lay if s in heads}
        dd = L.nest_dd(hh, {s: lay[s] for s in hh}, ms_list) if hh else pd.Series(dtype=float)
        depth, inc = choose_sensor(pf, key, span)
        src = "window heads product"
        n_ok = int(sum(np.isfinite(dd.get(m, np.nan)) for m in inc.index))
        if n_ok < need and fallback_changes and key in fallback_changes:
            ch, lay2 = fallback_changes[key]
            dd = L.nest_dd_from_changes(ch, lay2, ms_list)
            src = "printed table-3 well columns (fallback)"
        world = ms["world"]["piles"][key]
        minc = dict(zip(world["months"], world["inc_cm"], strict=True))
        rows = [(m, inc[m], dd.get(m, np.nan), dd.get(L.prev_month(m), np.nan),
                 dd.get(L.prev_month(m, 2), np.nan), minc.get(m, np.nan)) for m in inc.index]
        rows = [r for r in rows if np.isfinite(r[2])]
        per[key] = {"depth_m": depth, "n_months": len(rows), "head_source": src,
                    "pile_months_verified": int(len(inc))}
        for _m, c, d0, d1, d2, mm in rows:
            ys.append(c)
            xs.append(d0)
            x1.append(d1)
            x2.append(d2)
            mi.append(mm)
            gs.append(key)
    gate_obs = all(per[k]["n_months"] >= need for k in L.PRIMARY)
    gate_model = bool(ms["informative_model_side_pass"])
    out = {"role": role, "g_mod": g_mod, "piles": per, "informative": gate_obs and gate_model}
    if not out["informative"]:
        why = ("model-side gate failed" if not gate_model else
               f"fewer than {need} months with a verified pile value and a nest dD at a "
               "primary pile")
        out.update({"verdict": "not_informative", "why": why,
                    "wording": WORDING["not_informative"].format(why=why)})
        return out
    y, x, g = np.asarray(ys), np.asarray(xs), np.asarray(gs)
    est = L.conversion_estimate(y, x, g)
    v = L.conversion_verdict(est["g"], est["lo"], est["hi"], g_mod)
    out.update({"estimate": est, **v})
    if v["rho"] is not None and v["rho_ci"] is not None:
        out["wording"] = WORDING[v["verdict"]].format(rho=v["rho"], lo=v["rho_ci"][0],
                                                       hi=v["rho_ci"][1])
    else:
        out["wording"] = (f"compaction per metre of head change at the 2 Yunlin pile cells: "
                          f"{v['verdict'].replace('_', ' ')} (estimate or interval not finite)")
    out["report_only"] = L.report_only_lines(y, x, np.asarray(x1), np.asarray(x2), g,
                                             np.asarray(mi), g_mod)
    gp = (ms.get("g_mod_hindcast_paired") or {}).get("g_mod")
    if gp and np.isfinite(gp) and np.isfinite(est["g"]):
        out["report_only"]["rho_paired_hindcast"] = {
            "g_mod_hindcast_paired": float(gp), "rho": float(est["g"] / gp),
            "label": "g_obs over the column's slope on the hindcast heads that drove it "
                     "(2016-06..2022-12); the decision's g_mod is per metre of DISPLAYED "
                     "(restarted) window heads (column pairing, report-only)"}
    if role != "gating":
        out["label"] = "report-only (role fixed by G0 before registration); C31 stays N"
    return out


def level(ms: dict, pf: pd.DataFrame, competitor: pd.DataFrame, span) -> dict:
    col = "competitor_12m_cm" if span == L.WINDOW else "competitor_24m_cm"
    zr = "zr_12m_cm" if span == L.WINDOW else "zr_24m_cm"
    obs, model, lev, det = {}, {}, {}, {}
    for key in L.PRIMARY:
        depth, _ = choose_sensor(pf, key, span)
        t = window_total(pf, key, depth, span)
        det[key] = {"depth_m": depth, **t}
        obs[key] = t["value"]
        model[key] = float(ms["zr_level"][key][zr])
        c = competitor[competitor["pile_key"] == key]
        lev[key] = float(c[col].iloc[0]) if len(c) and pd.notna(c[col].iloc[0]) else None
    v = L.level_verdict(obs, model, lev)
    v["observed_detail"] = det
    v["report_only_f08"] = L.level_verdict(
        obs, {k: 0.8 * m for k, m in model.items()}, lev)["verdict"]
    why = f"missing at {v.get('missing')}" if v["verdict"] == "not_evaluable" else ""
    v["wording"] = WORDING[v["verdict"]].format(why=why)
    return v


def xizhou_depth_ok(rows: pd.DataFrame, months: list[str]) -> pd.Series:
    """Per month: 溪州 is printed at the declared 300 m and at no other depth (rows of any
    flag and any depth, so a sensor printed at another depth drops the season on its own)."""
    spec = L.PILES["xizhou"]
    r = rows[rows["pile"].astype(str) == spec["pile"]]
    out = {}
    for m in months:
        dep = set(pd.to_numeric(r.loc[r["month"].astype(str) == m, "depth_m"],
                                errors="coerce").dropna().astype(int))
        out[m] = dep == {int(spec["depth_m"])}
    return pd.Series(out, dtype=bool)


def xizhou(ms: dict, mapping: dict, pf: pd.DataFrame, heads: dict,
           rows: pd.DataFrame | None = None) -> dict:
    xz = ms.get("xizhou_r1c") or {}
    nest = mapping["piles"]["xizhou"]["nest"]
    lay = {w["sid"]: w["layer"] for w in nest}
    hh = {s: heads[s] for s in lay if s in heads}
    ml = L.months(L.REF_MONTH, L.WINDOW[1])
    dd = L.nest_dd(hh, {s: lay[s] for s in hh}, ml) if hh else pd.Series(dtype=float)
    g = pf[pf["pile"] == L.PILES["xizhou"]["pile"]].drop_duplicates(["month", "depth_m"])
    inc = g[g["depth_m"] == L.PILES["xizhou"]["depth_m"]].set_index("month")["inc_cm"]
    depth_ok = xizhou_depth_ok(rows if rows is not None else pf, ml)
    thr = float(xz.get("discontinuity_threshold_cm") or L.DISCONTINUITY_MIN_CM)
    sc = L.season_coefficients(inc, dd, [2027], threshold_cm=thr, depth_ok=depth_ok)
    k_obs, k_mod = sc["pooled"]["k"], xz.get("k_mod_2027")
    kp = (xz.get("k_mod_hindcast_paired") or {}).get("k")
    return {"role": "report_only", "seasons": sc, "k_obs": k_obs, "k_mod": k_mod,
            "ratio_obs_over_mod": (k_obs / k_mod) if (k_obs is not None and k_mod) else None,
            "k_mod_definition": xz.get("k_mod_definition"),
            "ratio_obs_over_hindcast_paired": (k_obs / kp) if (k_obs is not None and kp)
            else None,
            "rehearsal": xz.get("rehearsal"),
            "label": "report-only; C13c/C27 wording only; no Changhua subsidence lever"}


# ----------------------------------------------------------------------------- runner
def _write_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=str)
        f.write("\n")


def _load_registered(prereg_dir, rows_dir) -> dict:
    from harness import prereg3

    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    doc, _ = prereg3.load_prereg(L.ID, **kw)
    return doc


def _files(doc: dict) -> tuple[dict, dict]:
    """The registered mapping.json and model_side.json (hashes checked by assert_can_score)."""
    f = doc["model_side"]["files"]
    mp, ms = (json.loads((REPO / f[n]["path"]).read_text(encoding="utf-8"))
              for n in ("mapping.json", "model_side.json"))
    return mp, ms


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    from common import guard
    from harness import prereg3

    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    out = Path(a.out) / "report.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the report is written once")
    h = int(a.horizon)
    span = L.WINDOW if h == 12 else L.WINDOW24
    nb, nl = (L.NOT_BEFORE, L.NO_LATER_THAN) if h == 12 else (L.NOT_BEFORE24,
                                                              L.NO_LATER_THAN24)
    now = L._now()
    if now < nb:
        raise ScoringRefused(f"not before {nb:%Y-%m-%d}")
    doc = _load_registered(prereg_dir, rows_dir)
    rep = {"prereg_id": L.ID, "horizon_months": h, "scorer_version": SCORER_VERSION,
           "scored_at": now.isoformat(), "roles": doc["roles"]}
    ok_p, why_p = product_grants(a.piles_product)
    ok_h, why_h = product_grants(a.heads_product)
    rep["products"] = {"piles": why_p, "heads": why_h}
    if now > nl or not (ok_p and ok_h):
        why = "after the no_later_than date" if now > nl else "no product grants " \
                                                             "prospective_scorer"
        if now <= nl and not getattr(a, "final", False):
            raise ScoringRefused(f"{why}: build the products first, or pass --final to "
                                 "write the not-evaluable record (nothing read, no look)")
        rep.update({"verdict": "not_evaluable", "why": why})
        _write_once(out, rep)
        return rep
    part = getattr(a, "part", "piles")
    if part == "xizhou" and h != 12:
        raise ScoringRefused("the 溪州 arm is a 12-month line only")
    order = order_status(pdir, a.n2c_report, need_step2=(part == "xizhou"))
    rep["order"], rep["part"] = order, part
    if not order["ok_piles"]:
        raise ScoringRefused(f"session order: N2's score look and N2C's step-(1) look must "
                             f"be logged first ({order})")
    if part == "xizhou" and not order["ok_xizhou"]:
        raise ScoringRefused("session order: the 溪州 arm needs N2C's step-(2) report")
    if part == "piles":
        tests = [L.TEST_CONV, L.TEST_LEVEL] if h == 12 else [L.TEST_CONV24, L.TEST_LEVEL24]
    else:
        tests = [L.TEST_XZ]
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    rep["post_registration_looks"] = {
        t: prereg3.assert_can_score(L.ID, L.WP, test_id=t, **kw)["post_registration_looks"]
        for t in tests}
    mapping, ms = _files(doc)
    piles = guard.open_product(a.piles_product, "prospective_scorer")
    raw = piles.read_table(PILE_TABLE)
    pf = pile_frame(raw)
    keys = L.PRIMARY if part == "piles" else ("xizhou",)
    ids = sorted({w["sid"] for k in keys for w in mapping["piles"][k]["nest"]})
    hp = guard.open_product(a.heads_product, "prospective_scorer")
    heads = heads_series(hp.read_table(a.heads_table), ids)
    if part == "xizhou":
        rep["T_xizhou"] = xizhou(ms, mapping, pf, heads, rows=raw)
    else:
        fb = None
        if WELL_TABLE in piles.outputs and a.stations:
            fb = fallback_changes(piles.read_table(WELL_TABLE), mapping, Path(a.stations))
        rep["T_conversion"] = conversion(ms, mapping, pf, heads, span, doc["roles"][tests[0]],
                                         fb)
        comp = guard.open_product(REPO / doc["leveling_competitor"]["dir"], "baseline")
        rep["T_level"] = level(ms, pf, comp.read_table("competitor.csv"), span)
    _write_once(out, rep)
    return rep


def fallback_changes(wells: pd.DataFrame, mapping: dict, stations: Path) -> dict:
    """{pile key: (changes, layers)} from the accepted printed well columns."""
    st = pd.read_parquet(stations) if stations.suffix == ".parquet" else pd.read_csv(stations)
    st["sid"] = st["sid"].astype(str)
    layer_of = {w["sid"]: w["layer"] for k in mapping["piles"]
                for w in mapping["piles"][k]["nest"]}
    for r in st.itertuples():
        if r.sid not in layer_of and str(getattr(r, "layer_code", "")).isdigit():
            layer_of[r.sid] = int(r.layer_code)
    out = {}
    for key in L.PRIMARY:
        site = mapping["piles"][key]["site"]
        pile = L.PILES[key]["pile"]
        sub = wells[wells["pile"].map(WC.normalise) == WC.normalise(pile)]
        sid_of = WC.map_wells(sub["well"], st, (site["x"], site["y"]))
        out[key] = WC.changes_from_printed(sub.assign(pile=pile), pile, sid_of, layer_of)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--horizon", choices=("12", "24"), default="12")
    s.add_argument("--part", choices=("piles", "xizhou"), default="piles",
                   help="piles: T-conversion and T-level (session step 11); xizhou: the "
                        "report-only arm (step 12, after N2C's step-(2) report)")
    s.add_argument("--piles-product", type=Path, required=True)
    s.add_argument("--heads-product", type=Path, required=True)
    s.add_argument("--heads-table", default="heads_monthly.parquet")
    s.add_argument("--stations", default=None, help="station metadata (sid, WellName, "
                                                    "x_twd97, y_twd97, layer_code)")
    s.add_argument("--n2c-report", type=Path, default=None)
    s.add_argument("--final", action="store_true",
                   help="write a not-evaluable record when no product grants "
                        "prospective_scorer by the deadline")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = run_score(a)
    print(json.dumps({k: rep.get(k) for k in ("verdict", "why")}, ensure_ascii=False))
    for k in ("T_conversion", "T_level", "T_xizhou"):
        if k in rep:
            print(k, rep[k].get("verdict", rep[k].get("status")), rep[k].get("wording", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
