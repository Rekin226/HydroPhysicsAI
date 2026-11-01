"""Rank 12: after-the-fact pre-origin pile conversion diagnostic, capped at N (lead runs it).

Run strictly AFTER N3.piles_prospective.v1 is registered, so it cannot shape N3's G0, bands
or page rule. Label: 'after-the-fact on a spent site; cannot promote'. Use:
``after_the_fact_analysis`` (checked against the 2026-10-18 ruling (b): pre-origin printed
piles are validator and after_the_fact_analysis only; deep_benchmarks v2 grants it and it is
gate-exempt; never design_time_evidence). No correction is applied; no Changhua subsidence
lever is shown; the outcome reaches only C27 and C13c wording (user ruling 2026-11-01,
ranks 9/12: README/Limits may cite it, capped at N).

* Yunlin (土庫國中, 中科虎尾): deep_benchmarks v2 feed, pre-origin variant, read through
  common.guard with window_end 2022-12 (土庫's 400 m feed TKJS_283_29 is defective, so its
  300 m sensor TKJS(1); 中科虎尾 NTUH(1) 300 m); never printed rows; flags 0-1 (the
  product's step catalogue, steps.csv, removed steps carry flag 1; 2, 8, 9 excluded). Nest
  heads: heads/L1_clean v3 (window_end 2022-12, flags 0-1) at N3's mapped nest wells. The
  estimator is N3's (n3lib fe_slope, moving-block bootstrap, N3's nest rule), on 2016-06..
  2022-12; the model world is the page model's hindcast at the same cells, wells and months
  (its column on the free-run heads it was fitted on). rho_pre = g_obs / g_mod; k = 1/rho.
* 溪州國小 300 m (Changhua): the printed 007001 values from the current release product the
  2026-10-18 ruling narrowed (model_ready/release_20261012b2/preorigin/piles_printed v1:
  validator and after_the_fact_analysis only; data months to 2022-12 published by 2022-12;
  ``value`` = compression, already converted from the printed sign by the 007001
  transcription), window_end 2022-12, flag 0, seasons 2019-2022, N3's season-aggregate
  estimator against R1c's hindcast world at the same cell and nest (t-interval df = n - 1).
  Never the upstream deep_benchmarks_printed v2 (its MANIFEST still lists baseline and
  design_time_evidence).

Gates: N3's G0 (its rank-12 block: Yunlin hindcast gate, R1c season gate) must pass for a
site, else that site's line is 'coefficient not identifiable from held piles' and none of
its values is read. The pile set is fixed in the DECLARATION from metadata only (judge: 'from
metadata only, first confirm that each primary pile has at least 24 pre-origin months;
otherwise declare n = 1'): deep_benchmarks v2 mapping.csv first_month / last_month of the
feed sensors (an upper bound of the 2016-06..2022-12 months), and for 溪州 the seasons
2019-22 not hit by the release product's dropped months (validation.json). At run time,
after the look, a declared pile with fewer than 24 joint months is dropped (declared now);
no pile left, or fewer than 3 kept 溪州 seasons, is 'not evaluable'. One run.

Usage (repository root, the lead, after N3 is registered):
    python -I dataprep_staging/n3_piles/posthoc_pile_conversion.py declare --validate|--write
        --n3-mapping-dir <dir> --g0-result <file> [--expect-self-sha <sha>]
    python -I dataprep_staging/n3_piles/posthoc_pile_conversion.py run
        --out results/dataprep/reports/N3_posthoc
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
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
USE = "after_the_fact_analysis"
WINDOW_END = "2022-12"
YUNLIN_SPAN = ("2016-06", "2022-12")
XZ_SEASONS = (2019, 2020, 2021, 2022)
MIN_MONTHS_PRE = 24
MIN_SEASONS = 3
FEED = "results/dataprep/deep_benchmarks/v2"
PRINTED = "results/dataprep/model_ready/release_20261012b2/preorigin/piles_printed/v1"
PRINTED_UPSTREAM = "results/dataprep/deep_benchmarks_printed/v2"
HEADS = "results/dataprep/heads/L1_clean/v3"
MECHANISM = "results/twin/mechanism/mechanism.json"
RULINGS_1021 = "results/dataprep/rulings/user_rulings_2026-10-21.json"
UNIT = "N3PH:preorigin_piles_2016-06_2022-12"
SITE_WORDS = ("SJES", "溪州", "TKJS", "土庫", "NTUH", "虎尾")
LABEL = "after-the-fact on a spent site; cannot promote"
WORDING = {
    "yunlin": "C27: on pre-origin piles (after-the-fact, spent sites) the column's compaction "
              "per metre of head change is {k:.2f} times observed (90 % {lo:.2f}-{hi:.2f}); "
              "no correction is applied",
    "yunlin_unbounded": "C27: on pre-origin piles (after-the-fact, spent sites) the column's "
                        "compaction per metre of head change is {k:.2f} times observed (90 % "
                        "interval unbounded: the observed slope's interval includes 0); no "
                        "correction is applied",
    "yunlin_undefined": "C27: on pre-origin piles (after-the-fact, spent sites) the ratio is "
                        "not defined (observed slope {verdict}); no correction is applied",
    "xizhou": "C13c: R1c's Changhua elastic coefficient is {k:.2f} times the recovery-season "
              "coefficient at 溪州 (after-the-fact, spent site); no Changhua subsidence lever",
    "not_identifiable": "coefficient not identifiable from held piles (N3 G0)",
    "not_evaluable": "not evaluable ({why})",
}


def site_looks(prereg_dir=None) -> list[dict]:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    p = d / "looks.jsonl"
    out = []
    for lk in prereg3.read_chained(p) if p.exists() else []:
        s = json.dumps({k: lk.get(k) for k in ("units", "data_scope")}, ensure_ascii=False)
        if any(w in s for w in SITE_WORDS):
            out.append({"time": lk.get("time"), "wp": lk.get("wp"),
                        "purpose": lk.get("purpose"), "data_scope": lk.get("data_scope"),
                        "units": (lk.get("units") or [])[:6]})
    return out


def _rec(p) -> dict:
    p = Path(p)
    p = p if p.is_absolute() else REPO / p
    return {"path": rel_to_repo(p), "sha256": L.sha256(p) if p.is_file() else None}


def g0_lines(g0: dict) -> dict:
    r = (g0 or {}).get("rank12_report_only") or {}
    return {"yunlin": bool((r.get("yunlin_hindcast_2016_06_2022_12") or {}).get("pass")),
            "xizhou": bool((r.get("xizhou_r1c") or {}).get("pass"))}


def _span_months(first, last, span=YUNLIN_SPAN) -> int:
    """Months of ``span`` inside [first, last] (an upper bound: gaps are not visible)."""
    if not isinstance(first, str) or not isinstance(last, str):
        return 0
    a, b = max(first[:7], span[0]), min(last[:7], span[1])
    return len(L.months(a, b)) if a <= b else 0


def metadata_inventory(feed_mapping: Path | None = None,
                       printed_validation: Path | None = None) -> dict:
    """METADATA ONLY, before the declaration: the feed sensors' first/last months
    (deep_benchmarks v2 mapping.csv; no value column is read) and the release product's
    dropped months (validation.json). Fixes the declared pile set (judge: n = 1 if only one
    primary pile has 24 pre-origin months) and the 溪州 seasons' upper bound."""
    fm = Path(feed_mapping or REPO / FEED / "mapping.csv")
    mp = pd.read_csv(fm, usecols=["sensor_id", "first_month", "last_month"])
    piles = {}
    for key in L.PRIMARY:
        sid = L.PILES[key]["feed_preorigin"]
        r = mp[mp["sensor_id"] == sid]
        n = _span_months(*(r[["first_month", "last_month"]].iloc[0].tolist())) if len(r) else 0
        piles[key] = {"sensor": sid, "span_months_upper_bound": n,
                      "declared": bool(n >= MIN_MONTHS_PRE)}
    pv = Path(printed_validation or REPO / PRINTED / "validation.json")
    dropped = []
    if pv.is_file():
        v = json.loads(pv.read_text(encoding="utf-8"))
        dropped = sorted(map(str, (v.get("pile_publication") or {}).get("dropped_months")
                             or []))
    seasons = [y for y in XZ_SEASONS
               if not any(m[:4] == str(y) and 6 <= int(m[5:7]) <= 11 for m in dropped)]
    declared = [k for k in L.PRIMARY if piles[k]["declared"]]
    return {"source": {"feed_mapping": _rec(fm)["path"], "feed_mapping_sha256": L.sha256(fm),
                       "printed_validation": _rec(pv)},
            "yunlin": piles, "yunlin_declared": declared, "n_yunlin": len(declared),
            "xizhou": {"seasons_upper_bound": seasons, "dropped_months": dropped,
                       "evaluable_upper_bound": len(seasons) >= MIN_SEASONS},
            "rule": "a primary pile is declared if its feed sensor spans >= 24 months of "
                    "2016-06..2022-12 (metadata); n = the declared piles (n = 1 allowed); at "
                    "run time a declared pile with < 24 joint months is dropped"}


def _manifest_rec(product: str) -> dict:
    m = REPO / product / "MANIFEST.json"
    if not m.is_file():
        return {"path": f"{product}/MANIFEST.json", "sha256": None}
    d = json.loads(m.read_text(encoding="utf-8"))
    return {"path": f"{product}/MANIFEST.json", "sha256": L.sha256(m),
            "allowed_uses": d.get("allowed_uses")}


def declaration(n3_mapping_dir: Path, g0_result: Path, prereg_dir=None,
                inventory: dict | None = None) -> dict:
    g0 = json.loads(Path(g0_result).read_text(encoding="utf-8"))
    inv = inventory if inventory is not None else metadata_inventory()
    files = {n: _rec(Path(n3_mapping_dir) / n) for n in ("mapping.json", "model_side.json")}
    code = {f"{L.E}/{n}": L.sha256(HERE / n) for n in ("posthoc_pile_conversion.py",
                                                        "n3lib.py")}
    code[L.SHARED_SCORER] = L.sha256(REPO / L.SHARED_SCORER)
    src = inv.get("source") or {}
    meta = {src["feed_mapping"]} if src.get("feed_mapping") else set()
    if (src.get("printed_validation") or {}).get("sha256"):
        meta.add(src["printed_validation"]["path"])
    inputs = sorted(set(code) | {r["path"] for r in files.values()}
                    | {_rec(g0_result)["path"]} | meta)
    mech = _rec(MECHANISM)
    return {
        "id": L.PH_ID, "wp": L.WP, "kind": "declaration",
        "title": "After-the-fact pre-origin pile conversion diagnostic (Yunlin feed piles + "
                 "溪州 printed), capped at N",
        "label": LABEL, "use": USE, "n3_mapping": files,
        "use_check": {
            "ruling": "2026-10-18 (b): pre-origin printed piles validator and "
                      "after_the_fact_analysis only; design_time_evidence is NOT used",
            "feed": {**_manifest_rec(FEED), "use": USE,
                     "note": "deep_benchmarks v2 grants after_the_fact_analysis (gate-exempt)"},
            "printed": {**_manifest_rec(PRINTED), "use": USE,
                        "note": "the current release product the 2026-10-18 ruling narrowed "
                                "(validator, after_the_fact_analysis); never the upstream "
                                f"{PRINTED_UPSTREAM}, whose MANIFEST still lists baseline "
                                "and design_time_evidence"}},
        "sensors": {"土庫國中": "deep_benchmarks v2 feed TKJS(1) 300 m (400 m feed TKJS_283_29 "
                                "defective, flag 9)",
                    "中科虎尾": "deep_benchmarks v2 feed NTUH(1) 300 m",
                    "溪州國小": "model_ready/release_20261012b2/preorigin/piles_printed v1, "
                                "300 m, flag 0 (value = compression)"},
        "inventory": inv,
        "declared_sets": {"yunlin_piles": inv["yunlin_declared"], "n_yunlin": inv["n_yunlin"],
                          "xizhou_seasons_upper_bound": inv["xizhou"]["seasons_upper_bound"]},
        "months": {"yunlin": list(YUNLIN_SPAN), "xizhou_seasons": list(XZ_SEASONS),
                   "never_opened": "any row after 2022-12"},
        "estimator": {"yunlin": "n3lib.fe_slope / conversion_estimate (N3's), nest rule "
                                "n3lib.nest_dd, model world = the page model's hindcast",
                      "xizhou": "n3lib.season_coefficients (N3's), model world = R1c's "
                                "hindcast", "code": code},
        "gates": {"g0": {"result": _rec(g0_result), "lines": g0_lines(g0)},
                  "min_months_yunlin": MIN_MONTHS_PRE, "min_seasons_xizhou": MIN_SEASONS,
                  "g0_failure": WORDING["not_identifiable"], "runs": "one"},
        "outcome_to_wording": {**WORDING, "status_cap": "N; C27 and C13c lines only; no "
                               "correction applied; no R1c-vs-Q0 separation",
                               "interval_includes_0": "the k interval is unbounded (rho's "
                               "interval includes 0): worded 'interval unbounded'"},
        "disclosures": {
            "prior_site_looks": site_looks(prereg_dir)[:150],
            "mechanism_json": {**mech, "span": "MLCW rings 2010-2025 (includes post-origin "
                                               "months); seasonal compaction per metre "
                                               "computed at the ring sites, same quantity, "
                                               "same schools"},
            "r1c_suspension": f"user ruling 2026-10-21 ({RULINGS_1021}): R1c's six-zone column "
                              "suspended as a subsidence fallback; its post-2022 uplift is an "
                              "artefact",
            "after_n3": f"declared after {L.ID} was registered; it cannot alter N3"},
        "zones": NA("no fit"), "priors": NA("no fit"), "burn_in": NA("no fit"),
        "folds": NA("after the fact"), "tests": NA("declaration; no verdict gates anything"),
        "fresh_units": NA("spent sites; no fresh unit"),
        "thresholds": {"ratio_band": L.RATIO_BAND, "level": L.LEVEL, "boot_B": L.BOOT_B,
                       "boot_block": L.BOOT_BLOCK, "min_months": MIN_MONTHS_PRE,
                       "min_seasons": MIN_SEASONS},
        "gate_scores": NA("declaration"),
        "primary_model": {"yunlin": "page model (datum_gate_innov)",
                          "xizhou": "R1c (datum_gate_ccz)"},
        "inputs": inputs,
        "interpretation": "Wording only (C27, C13c), capped at N; after-the-fact on spent "
                          "sites; nothing is promoted or corrected.",
        "written_by": f"{L.E}/posthoc_pile_conversion.py"}


# ----------------------------------------------------------------------------- run
def _feed_rows(use_cols) -> pd.DataFrame:
    from common import guard

    p = guard.open_product(REPO / FEED, USE, window_end=WINDOW_END)
    df = p.read_table("monthly.parquet", columns=list(use_cols) + ["variant"])
    return df[df["flag"].isin([0, 1]) & (df["variant"] == "preorigin")]


def _printed_rows(use_cols) -> pd.DataFrame:
    from common import guard

    p = guard.open_product(REPO / PRINTED, USE, window_end=WINDOW_END)
    df = p.read_table("printed_monthly.parquet", columns=use_cols)
    return df[df["flag"] == 0]


def _heads_rows(ids) -> dict[str, pd.Series]:
    from common import guard

    p = guard.open_product(REPO / HEADS, USE, window_end=WINDOW_END)
    df = p.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag"])
    df = df[df["flag"].isin([0, 1]) & df["sid"].astype(str).isin(ids)]
    df["month"] = df["date"].astype(str).str[:7]
    return {str(s): g.groupby("month")["value"].mean() for s, g in df.groupby("sid")}


def inverse_interval(ci) -> list[float] | None:
    """1 / [a, b]: [1/b, 1/a] when the interval excludes 0; None (unbounded) when it
    includes 0 or is missing."""
    if ci is None or not all(np.isfinite(ci)):
        return None
    a, b = sorted(float(x) for x in ci)
    if a <= 0.0 <= b:
        return None
    return sorted([1.0 / a, 1.0 / b])


def yunlin(ms: dict, mapping: dict, piles=L.PRIMARY) -> dict:
    """The declared Yunlin piles (n = 1 allowed); a pile with fewer than MIN_MONTHS_PRE joint
    months is dropped (declared rule); none left: not evaluable."""
    piles = tuple(piles)
    if not piles:
        return {"line": "not_evaluable", "wording": WORDING["not_evaluable"].format(
            why="no Yunlin pile has 24 pre-origin months in the metadata (declared n = 0)")}
    feed = _feed_rows(["sensor_id", "month", "value", "flag"])
    ids = sorted({w["sid"] for k in piles for w in mapping["piles"][k]["nest"]})
    heads = _heads_rows(ids)
    rows = {k: [] for k in piles}
    for key in piles:
        s = feed[feed["sensor_id"] == L.PILES[key]["feed_preorigin"]]
        inc = s.drop_duplicates("month").set_index("month")["value"].astype(float)
        nest = mapping["piles"][key]["nest"]
        lay = {w["sid"]: w["layer"] for w in nest}
        hh = {k: v for k, v in heads.items() if k in lay}
        dd = L.nest_dd(hh, {k: lay[k] for k in hh}, L.months(*YUNLIN_SPAN))
        w = ms["hindcast"]["piles"][key]
        mi = dict(zip(w["months"], w["inc_cm"], strict=True))
        md = dict(zip(w["months"], w["dd_nest_m"], strict=True))
        for m in L.months(*YUNLIN_SPAN):
            if m in inc.index and np.isfinite(dd.get(m, np.nan)) and np.isfinite(
                    mi.get(m, np.nan)) and np.isfinite(md.get(m, np.nan)):
                rows[key].append((inc[m], dd[m], mi[m], md[m]))
    n = {k: len(v) for k, v in rows.items()}
    kept = [k for k in piles if n[k] >= MIN_MONTHS_PRE]
    if not kept:
        why = f"fewer than {MIN_MONTHS_PRE} joint months at every declared pile ({n})"
        return {"line": "not_evaluable", "n": n,
                "wording": WORDING["not_evaluable"].format(why=why)}
    ys, xs, mys, mxs, gs = [], [], [], [], []
    for k in kept:
        for y, x, my, mx in rows[k]:
            ys.append(y)
            xs.append(x)
            mys.append(my)
            mxs.append(mx)
            gs.append(k)
    g = np.asarray(gs)
    g_mod = L.fe_slope(mys, mxs, g)
    est = L.conversion_estimate(np.asarray(ys), np.asarray(xs), g)
    v = L.conversion_verdict(est["g"], est["lo"], est["hi"], g_mod)
    out = {"line": "computed", "n": n, "piles_used": kept, "n_piles": len(kept),
           "dropped_at_run": [k for k in piles if k not in kept], "g_mod_pre": g_mod,
           "estimate": est, **v, "label": LABEL}
    rho = v.get("rho")
    if rho is None or rho == 0:
        out.update({"k_model_over_obs": None, "k_ci90": None,
                    "wording": WORDING["yunlin_undefined"].format(verdict=v["verdict"])})
        return out
    k, ci = 1.0 / rho, inverse_interval(v.get("rho_ci"))
    out.update({"k_model_over_obs": k, "k_ci90": ci,
                "wording": (WORDING["yunlin"].format(k=k, lo=ci[0], hi=ci[1]) if ci else
                            WORDING["yunlin_unbounded"].format(k=k))})
    return out


def xizhou_line(ms: dict, mapping: dict) -> dict:
    pr = _printed_rows(["station", "depth_m", "month", "value", "flag"])
    x = pr[(pr["station"] == L.PILES["xizhou"]["pile"]) & (pr["depth_m"] == 300)]
    inc = x.drop_duplicates("month").set_index("month")["value"].astype(float) / 10.0
    nest = mapping["piles"]["xizhou"]["nest"]
    lay = {w["sid"]: w["layer"] for w in nest}
    heads = _heads_rows(sorted(lay))
    dd = L.nest_dd(heads, {k: lay[k] for k in heads}, L.months("2019-05", "2022-12"))
    w = ms["xizhou_r1c"]["hindcast"]["piles"]["xizhou"]
    minc = pd.Series(w["inc_cm"], index=w["months"])
    mdd = pd.Series(w["dd_nest_m"], index=w["months"])
    thr = L.discontinuity_threshold(minc.to_numpy())
    obs = L.season_coefficients(inc, dd, XZ_SEASONS, threshold_cm=thr)
    mod = L.season_coefficients(minc, mdd, XZ_SEASONS, threshold_cm=thr)
    ko, km = obs["pooled"]["k"], mod["pooled"]["k"]
    if obs["pooled"]["n"] < MIN_SEASONS or ko is None or not km:
        why = f"fewer than {MIN_SEASONS} kept seasons"
        return {"line": "not_evaluable", "obs": obs, "model": mod,
                "wording": WORDING["not_evaluable"].format(why=why)}
    k = km / ko
    return {"line": "computed", "obs": obs, "model": mod, "k_model_over_obs": k,
            "label": LABEL, "wording": WORDING["xizhou"].format(k=k)}


def run(out: Path, prereg_dir=None, rows_dir=None) -> dict:
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    out = Path(out)
    rp = out / "report.json"
    if rp.exists():
        raise FileExistsError(f"{rp} exists: one run")
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    decl, sha = prereg3.load_prereg(L.PH_ID, **kw)
    if not any(r["id"] == L.ID for r in prereg3.read_chained(pdir / "INDEX.jsonl")):
        raise SystemExit(f"{L.ID} is not registered: this diagnostic runs only after it")
    bad = prereg3.changed_inputs(decl)
    if bad:
        raise SystemExit(f"registered inputs changed: {bad}")
    g0 = json.loads((REPO / decl["gates"]["g0"]["result"]["path"]).read_text(encoding="utf-8"))
    lines = g0_lines(g0)
    mapping, ms = (json.loads((REPO / decl["n3_mapping"][n]["path"]).read_text(
        encoding="utf-8")) for n in ("mapping.json", "model_side.json"))
    rep = {"prereg_id": L.PH_ID, "declaration_sha256": sha, "label": LABEL, "use": USE,
           "g0_lines": lines}
    if not any(lines.values()):
        rep["yunlin"] = rep["xizhou"] = {"line": "not_identifiable",
                                         "wording": WORDING["not_identifiable"]}
        rep["values_read"] = False
    else:
        keys = (list(decl["declared_sets"]["yunlin_piles"]) if lines["yunlin"] else []) \
            + (["xizhou"] if lines["xizhou"] else [])
        ids = sorted({L.PILES[k]["feed_preorigin"] for k in keys}
                     | {w["sid"] for k in keys for w in mapping["piles"][k]["nest"]})
        members = {UNIT: {"description": "pre-origin deep-pile months (Yunlin feed 2016-06.."
                                         "2022-12, 溪州 printed 2019-05..2022-12) and their "
                                         "nest heads, for the sites whose G0 line passed; "
                                         "after the fact on spent sites",
                          "ids": ids, "window": "2016-06..2022-12"}}
        prereg3.record_look([UNIT], "deep_benchmarks+heads:2016-06..2022-12", L.WP,
                            "after_the_fact", members=members, prereg_id=L.PH_ID,
                            note=LABEL, **kw)
        rep["values_read"] = True
        rep["yunlin"] = (yunlin(ms, mapping, decl["declared_sets"]["yunlin_piles"])
                         if lines["yunlin"] else
                         {"line": "not_identifiable", "wording": WORDING["not_identifiable"]})
        rep["xizhou"] = (xizhou_line(ms, mapping) if lines["xizhou"] else
                         {"line": "not_identifiable", "wording": WORDING["not_identifiable"]})
    out.mkdir(parents=True, exist_ok=True)
    with open(rp, "x", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, sort_keys=True, default=float)
        f.write("\n")
    return rep


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("declare")
    g = d.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    d.add_argument("--n3-mapping-dir", type=Path, required=True)
    d.add_argument("--g0-result", type=Path, required=True)
    d.add_argument("--expect-self-sha", default=None)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rep = run(a.out, prereg_dir, rows_dir)
        print(json.dumps({k: rep[k].get("wording") for k in ("yunlin", "xizhou")},
                         ensure_ascii=False, indent=1))
        return 0
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    doc = declaration(a.n3_mapping_dir, a.g0_result, prereg_dir)
    errs = prereg3.validate_prereg(doc, prereg_dir=pdir)
    print("validate:", errs or "ok", "| g0 lines:", doc["gates"]["g0"]["lines"],
          "| declared:", json.dumps(doc["declared_sets"], ensure_ascii=False))
    print("this script sha256:", L.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = list(errs)
    if not any(r["id"] == L.ID for r in prereg3.read_chained(pdir / "INDEX.jsonl")):
        bad.append(f"{L.ID} must be registered first")
    if a.expect_self_sha != L.sha256(Path(__file__)):
        bad.append("--expect-self-sha does not match this script")
    if bad:
        print("REFUSED:", bad)
        return 1
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(doc, **kw)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
