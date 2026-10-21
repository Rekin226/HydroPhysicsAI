"""X4 leveling products from the verified transcriptions of the WRA summary reports and
briefings (lev_extract.py), validated against the 1997-2021 leveling panel in hand.

    python -I lev_build.py <transcriptions.json> [--out-root <dir>]

Products (results/dataprep/subsidence_public_reports/{preorigin,postorigin}/v<N>/):

* ``county_intervals.csv``   per county (彰化, 雲林) and survey interval (表2-1, 表2-2): the
  maximum annual subsidence rate (cm/yr) and its township, the areas above 2 and 3 cm/yr
  and the rate-class areas (km2), consolidated over report editions;
* ``county_area_by_year.csv`` 表1-1 significant-subsidence area by county (km2);
* ``max_rate_by_region.csv``  表1-4 maximum annual rate by region (cm/yr; Yunlin coastal /
  inland);
* ``survey_dates.csv``        表1-2 leveling field dates (the window each rate refers to);
* ``township_status.csv``     per township and survey year: listed as exceeding 3 cm/yr or
  not (only where the source gives the list), the county maximum and its township;
* ``township_bounds.csv``     the same as an interval on the township's peak annual rate:
  lower bound 3 cm/yr when listed, upper bound the county maximum (or 3 cm/yr when not
  listed or when the county has no significant area), the value itself for the county's
  maximum-rate township;
* ``hsr_piers.csv``           (post-origin product) high-speed-rail pier settlement 2022-2025
  printed in the 114 briefing (Taiwan High Speed Rail data, cited by the WRA).

The pre-origin product holds survey years up to 2022 (ROC 111, May 2022); the post-origin
product (rule U7: report-only, never a gate, choice, hyperparameter or threshold) holds the
2023-2025 surveys. Every value is a printed number that two independent transcriptions
agree on; disagreements are kept in ``unverified.csv`` and never used.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import manifest  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402

from x4_canal_leveling.config import (  # noqa: E402
    DATAPREP,
    LEVELING_PANEL,
    PROD_LEVEL_POST,
    PROD_LEVEL_PRE,
)

TOWNS_JSON = REPO / "data_fetch_api" / "new_measurements" / "taiwan_atlas" / "towns-10t_2021.json"
COUNTY = {"彰化": "彰化縣", "雲林": "雲林縣"}
PANEL_ALIAS = {"台西鄉": "臺西鄉", "員林鎮": "員林市", "大埠鄉": "大埤鄉", "西湖鎮": "溪湖鎮"}
LAST_PRE_ROC = 111          # survey of May 2022 is the last pre-origin survey
#: Benchmark-pair rates above this are datum artefacts (resets, rebuilt marks), as in
#: hydrophysics/twin/leveling.py's max_rate screen; fixed before the comparison.
SCREEN_CM_YR = 15.0


def edition(item: dict) -> tuple[str, int | None]:
    name = Path(item["pdf"]).stem
    m = re.match(r"roc(\d{3})", name) or re.search(r"results_(\d{3})", name)
    if m:
        return item["group"], int(m[1])
    m = re.match(r"cy_(\d{4})-(\d{2})", name)
    return item["group"], (int(m[1]) - 1911) if m else None


def town_lists() -> dict[str, list[str]]:
    import geopandas as gpd

    tw = gpd.read_file(TOWNS_JSON, layer="towns")
    return {c: sorted(tw[c == tw.COUNTYNAME].TOWNNAME.tolist()) for c in COUNTY.values()}


def norm_town(name: str | None, county: str, towns: dict) -> str | None:
    """Printed township -> atlas 2021 name (prefix match inside the county)."""
    if not name:
        return None
    n = "".join(str(name).split()).replace("台", "臺")
    n = PANEL_ALIAS.get(n, n)
    lst = towns[COUNTY[county]]
    if n in lst:
        return n
    cand = [t for t in lst if t[:2] == n[:2]]
    return cand[0] if len(cand) == 1 else None


def load(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in d["pages"]:
        grp, ed = edition(p["item"])
        for r in p["rows"]:
            out.append({"group": grp, "edition_roc": ed, "kind": p["item"]["kind"],
                        "file": p["item"]["pdf"], "page": p["item"]["page"], **r})
    return out


# ----------------------------------------------------------------------------- tables
def interval_table(rows: list[dict], towns: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = []
    for r in rows:
        if r["kind"] not in ("interval_chc", "interval_yl") or not r["any_verified"]:
            continue
        cty, start, end = r["key"]
        if not cty or ("彰" not in cty and "雲" not in cty):
            continue                 # tables of other regions printed on the same page
        cty = "彰化" if "彰" in cty else "雲林"
        v = r["value"]
        try:
            end_roc, end_m = int(end.split(".")[0]), int(end.split(".")[1])
            start_roc, start_m = int(start.split(".")[0]), int(start.split(".")[1])
        except (ValueError, IndexError):
            continue
        recs.append({"county": cty, "start": start, "end": end,
                     "start_ym": f"{start_roc + 1911}-{start_m:02d}",
                     "end_ym": f"{end_roc + 1911}-{end_m:02d}", "survey_year": end_roc + 1911,
                     "edition_roc": r["edition_roc"], "file": r["file"], "page": r["page"],
                     "max_rate_cm_yr": v["max_rate_cm_yr"],
                     "max_rate_township_printed": v["max_rate_township"],
                     "max_rate_township": norm_town(v["max_rate_township"], cty, towns),
                     "area_gt2_km2": v["area_gt2_km2"], "area_gt3_km2": v["area_gt3_km2"],
                     "class_areas_km2": json.dumps(v["class_areas"], ensure_ascii=False)})
    d = pd.DataFrame(recs)
    # consolidate over editions by survey year (editions print the same interval as e.g.
    # 100.5~101.5 or 100.6~101.6): the latest edition's value; record the others' agreement
    cons = []
    for _key, g in d.groupby(["county", "survey_year"]):
        g = g.sort_values("edition_roc", ascending=False)
        top = g.iloc[0].to_dict()
        fields = ["max_rate_cm_yr", "max_rate_township", "area_gt2_km2", "area_gt3_km2",
                  "class_areas_km2"]
        dis = {f: sorted({int(x.edition_roc) for x in g.itertuples()
                          if _neq(getattr(x, f), top[f])}) for f in fields}
        top["editions"] = ",".join(str(int(x)) for x in g.edition_roc)
        top["n_editions"] = len(g)
        top["editions_disagreeing"] = json.dumps({k: v for k, v in dis.items() if v})
        cons.append(top)
    return pd.DataFrame(cons).sort_values(["county", "end_ym"]), d


def _neq(a, b) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return not (b is None or (isinstance(b, float) and np.isnan(b)))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > 1e-9
    return a != b


def simple_table(rows: list[dict], kind: str, cols: list[str]) -> pd.DataFrame:
    recs = []
    for r in rows:
        if r["kind"] != kind or not r["any_verified"]:
            continue
        recs.append({**{c: k for c, k in zip(cols, r["key"], strict=True)}, **r["value"],
                     "edition_roc": r["edition_roc"], "file": r["file"], "page": r["page"]})
    return pd.DataFrame(recs)


def statements(rows: list[dict], towns: dict) -> pd.DataFrame:
    recs = []
    for r in rows:
        if r["kind"] not in ("statement", "brief_statement") or not r["any_verified"]:
            continue
        cty, yr = r["key"]
        if not cty or yr is None or ("彰" not in cty and "雲" not in cty):
            continue
        cty = "彰化" if "彰" in cty else "雲林"
        v = r["value"]
        list_ok = v["townships_over_3cm"] is not None and v["says_none_over_3cm"] is not None
        listed = [norm_town(t, cty, towns) for t in (v["townships_over_3cm"] or [])]
        recs.append({"county": cty, "survey_roc": int(yr), "survey_year": int(yr) + 1911,
                     "group": r["group"], "edition_roc": r["edition_roc"], "file": r["file"],
                     "page": r["page"],
                     "townships_over_3cm": json.dumps(sorted(t for t in listed if t),
                                                      ensure_ascii=False),
                     "unmatched_names": json.dumps(
                         [t for t, n in zip(v["townships_over_3cm"] or [], listed, strict=True)
                          if n is None], ensure_ascii=False),
                     "says_none_over_3cm": bool(v["says_none_over_3cm"]),
                     "list_verified": bool(list_ok),
                     "max_rate_cm_yr": v["max_rate_cm_yr"],
                     "max_rate_township": norm_town(v["max_rate_township"], cty, towns),
                     "significant_area_km2": v["significant_area_km2"]})
    return pd.DataFrame(recs)


def township_status(st: pd.DataFrame, iv: pd.DataFrame, towns: dict) -> pd.DataFrame:
    """One row per (county, survey year, township): the list status from each source that
    gives a list for the county and year, plus the county maximum of the interval table."""
    out = []
    for (cty, yr), g in st.groupby(["county", "survey_year"]):
        ivr = iv[(iv.county == cty) & (iv.survey_year == yr)]
        cmax = float(ivr.max_rate_cm_yr.iloc[0]) if len(ivr) else np.nan
        ctown = ivr.max_rate_township.iloc[0] if len(ivr) else None
        a3 = float(ivr.area_gt3_km2.iloc[0]) if len(ivr) and pd.notna(
            ivr.area_gt3_km2.iloc[0]) else np.nan
        g = g[g.list_verified]
        gives_list = g[(g.townships_over_3cm != "[]") | g.says_none_over_3cm.astype(bool)]
        if gives_list.empty:
            continue
        lists = [set(json.loads(x)) for x in gives_list.townships_over_3cm]
        union, inter = set().union(*lists), set.intersection(*lists)
        for t in towns[COUNTY[cty]]:
            n_src = len(lists)
            n_listed = sum(t in s for s in lists)
            out.append({"county": cty, "survey_year": int(yr), "township": t,
                        "n_sources_with_list": n_src, "n_sources_listing": n_listed,
                        "listed_by_all": t in inter, "listed_by_any": t in union,
                        "sources": ";".join(f"{a}{b}" for a, b in zip(
                            gives_list.group, gives_list.edition_roc, strict=True)),
                        "county_max_rate_cm_yr": cmax, "county_max_township": ctown,
                        "county_area_gt3_km2": a3, "is_max_township": t == ctown})
    return pd.DataFrame(out)


def bounds(ts: pd.DataFrame) -> pd.DataFrame:
    b = ts.copy()
    lo = np.where(b.listed_by_all, 3.0, np.nan)
    hi = np.where(b.listed_by_any, b.county_max_rate_cm_yr,
                  np.minimum(3.0, b.county_max_rate_cm_yr.fillna(3.0)))
    hi = np.where(b.county_area_gt3_km2.fillna(1.0) == 0.0,
                  np.minimum(3.0, b.county_max_rate_cm_yr.fillna(3.0)), hi)
    b["peak_rate_lower_cm_yr"] = lo
    b["peak_rate_upper_cm_yr"] = hi
    b["peak_rate_value_cm_yr"] = np.where(b.is_max_township, b.county_max_rate_cm_yr, np.nan)
    b["ambiguous"] = b.listed_by_any & ~b.listed_by_all
    b["note"] = np.where(b.ambiguous, "sources disagree on the listing: upper bound only", "")
    b["quantity"] = ("peak of the WRA interpolated annual subsidence-rate field (cm/yr, "
                     "positive = sinking) inside the township over the survey interval; the "
                     "upper bound of an unlisted township applies to the contoured field, not "
                     "to single benchmarks (see validation.json unlisted_summary)")
    return b


# ----------------------------------------------------------------------------- validation
def panel_rates(towns: dict) -> pd.DataFrame:
    df = pd.read_parquet(LEVELING_PANEL, columns=["datetime", "elev_m", "sid", "town"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["town"] = df["town"].map(lambda t: PANEL_ALIAS.get(t, t))
    c_of = {t: k for k, c in COUNTY.items() for t in towns[c]}
    df["county"] = df["town"].map(c_of)
    df = df.dropna(subset=["county"]).sort_values(["sid", "datetime"])
    out = []
    for sid, g in df.groupby("sid"):
        t = g.datetime.to_numpy()
        z = g.elev_m.to_numpy()
        for i in range(1, len(g)):
            dt = (t[i] - t[i - 1]) / np.timedelta64(1, "D") / 365.25
            if 0.6 <= dt <= 1.5 and 3 <= pd.Timestamp(t[i]).month <= 8:
                out.append({"sid": sid, "town": g.town.iloc[i], "county": g.county.iloc[i],
                            "survey_year": pd.Timestamp(t[i]).year,
                            "rate_cm_yr": (z[i - 1] - z[i]) / dt * 100.0})
    return pd.DataFrame(out)


def validate(iv: pd.DataFrame, st: pd.DataFrame, pr: pd.DataFrame) -> dict:
    res = {"max_rate": [], "towns": [], "area": {}}
    for (cty, yr), g in pr.groupby(["county", "survey_year"]):
        if yr < 2012 or yr > 2021:
            continue
        r = iv[(iv.county == cty) & (iv.survey_year == yr)]
        if r.empty:
            continue
        gs = g[g.rate_cm_yr.abs() <= SCREEN_CM_YR]
        i = gs.rate_cm_yr.idxmax()
        res["max_rate"].append({
            "county": cty, "survey_year": int(yr), "n_benchmarks": int(len(g)),
            "printed_max": float(r.max_rate_cm_yr.iloc[0]),
            "panel_max_raw": round(float(g.rate_cm_yr.max()), 2),
            "panel_max": round(float(gs.rate_cm_yr.max()), 2),
            "n_screened_out": int(len(g) - len(gs)),
            "panel_p99": round(float(gs.rate_cm_yr.quantile(0.99)), 2),
            "printed_town": r.max_rate_township.iloc[0], "panel_town": gs.town.loc[i],
            "panel_frac_gt3": round(float((gs.rate_cm_yr > 3).mean()), 4),
            "printed_area_gt3": float(r.area_gt3_km2.iloc[0]) if pd.notna(
                r.area_gt3_km2.iloc[0]) else None})
        s = st[(st.county == cty) & (st.survey_year == yr)]
        if len(s):
            listed = set().union(*[set(json.loads(x)) for x in s.townships_over_3cm])
            panel_t = set(gs[gs.rate_cm_yr > 3].town)
            tmax = gs.groupby("town").rate_cm_yr.max()
            unl = tmax[~tmax.index.isin(listed)]
            res.setdefault("unlisted_town_peaks", []).extend(
                {"county": cty, "survey_year": int(yr), "town": t, "panel_max": round(float(v), 2)}
                for t, v in unl.items())
            res["towns"].append({"county": cty, "survey_year": int(yr),
                                 "printed": sorted(listed), "panel_any_gt3": sorted(panel_t),
                                 "printed_not_in_panel": sorted(listed - panel_t),
                                 "panel_not_printed": sorted(panel_t - listed)})
    mr = pd.DataFrame(res["max_rate"])
    if len(mr):
        mr["diff"] = mr.panel_max - mr.printed_max
        res["max_rate_summary"] = {
            "n": int(len(mr)), "median_abs_diff_cm_yr": float(mr["diff"].abs().median()),
            "within_0.5": int((mr["diff"].abs() <= 0.5).sum()),
            "within_1.0": int((mr["diff"].abs() <= 1.0).sum()),
            "town_match": int((mr.printed_town == mr.panel_town).sum())}
        ok = mr.dropna(subset=["printed_area_gt3"])
        for cty, g in ok.groupby("county"):
            if len(g) >= 4:
                res["area"][cty] = {"n": int(len(g)), "spearman_area_vs_frac_gt3": float(
                    g.printed_area_gt3.rank().corr(g.panel_frac_gt3.rank()))}
    tw = res["towns"]
    if tw:
        tp = sum(len(set(x["printed"]) & set(x["panel_any_gt3"])) for x in tw)
        fp = sum(len(x["printed_not_in_panel"]) for x in tw)
        fn = sum(len(x["panel_not_printed"]) for x in tw)
        res["towns_summary"] = {"county_years": len(tw), "agree": tp,
                                "printed_only": fp, "panel_only": fn}
        up = pd.DataFrame(res.get("unlisted_town_peaks", []))
        if len(up):
            res["unlisted_summary"] = {
                "town_years_with_benchmarks": int(len(up)),
                "frac_benchmark_peak_gt3": float((up.panel_max > 3).mean()),
                "p90_benchmark_peak_cm_yr": float(up.panel_max.quantile(0.9)),
                "max_benchmark_peak_cm_yr": float(up.panel_max.max()),
                "note": "unlisted townships can hold single benchmarks above 3 cm/yr: the "
                        "printed lists name townships with a contoured area above 3 cm/yr"}
    return res


def consistency(iv_all: pd.DataFrame, area: pd.DataFrame, mrate: pd.DataFrame,
                iv: pd.DataFrame) -> dict:
    """Printed inconsistencies: across editions (same interval) and across tables."""
    out = {"cross_edition": [], "cross_table": []}
    for r in iv.itertuples():
        dis = json.loads(r.editions_disagreeing)
        if dis:
            out["cross_edition"].append({"county": r.county, "interval": f"{r.start}~{r.end}",
                                         "latest_edition": int(r.edition_roc),
                                         "fields": dis})
    if len(mrate):
        latest = mrate.sort_values("edition_roc").groupby(["region", "roc_year"]).tail(1)
        for r in iv.itertuples():
            reg = "彰化" if r.county == "彰化" else "雲林內陸"
            m = latest[(latest.region == reg) & (latest.roc_year == r.survey_year - 1911)]
            if len(m) and r.county == "彰化" and _neq(float(m.rate_cm_yr.iloc[0]),
                                                     r.max_rate_cm_yr):
                out["cross_table"].append({"county": r.county, "survey_year": r.survey_year,
                                           "table_1_4": float(m.rate_cm_yr.iloc[0]),
                                           "table_2_x": r.max_rate_cm_yr})
    if len(area):
        la = area.sort_values("edition_roc").groupby(["county", "year_label"]).tail(1)
        for r in iv.itertuples():
            lab = str(r.survey_year - 1911)
            a = la[(la.county == r.county) & (la.year_label == lab)]
            if len(a) and pd.notna(r.area_gt3_km2) and _neq(float(a.area_km2.iloc[0]),
                                                            float(r.area_gt3_km2)):
                out["cross_table"].append({"county": r.county, "survey_year": r.survey_year,
                                           "table_1_1_area": float(a.area_km2.iloc[0]),
                                           "table_2_x_area_gt3": float(r.area_gt3_km2)})
    return out


# ----------------------------------------------------------------------------- build
def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    tpath = Path(argv[0])
    root = Path(argv[argv.index("--out-root") + 1]) if "--out-root" in argv else DATAPREP
    towns = town_lists()
    rows = load(tpath)
    unverified = pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(
        v, (dict, list)) else v) for k, v in r.items()} for r in rows if not r["verified"]])
    unverified["note"] = "fields not in fields_verified=true are never used"
    iv, iv_all = interval_table(rows, towns)
    area = simple_table(rows, "area_county", ["county", "year_label"])
    mrate = simple_table(rows, "max_rate", ["region", "roc_year"])
    sdates = simple_table(rows, "survey_dates", ["region"])
    st = statements(rows, towns)
    ts = township_status(st, iv, towns)
    bd = bounds(ts)
    hsr = simple_table(rows, "hsr", ["chainage", "roc_year"])
    pr = panel_rates(towns)
    val = validate(iv, st, pr)
    cons = consistency(iv_all, area, mrate, iv)
    n_val = {"rows_total": len(rows), "rows_verified": int(sum(r["verified"] for r in rows)),
             "rows_partly_verified": int(sum(r["any_verified"] and not r["verified"]
                                             for r in rows)),
             "rows_unverified": int(sum(not r["any_verified"] for r in rows)),
             "fields_total": int(sum(len(r["fields_verified"]) for r in rows)),
             "fields_verified": int(sum(sum(r["fields_verified"].values()) for r in rows))}
    code = [Path(__file__), Path(__file__).with_name("lev_extract.py"),
            Path(__file__).with_name("lev_pages.py"), Path(__file__).with_name("config.py")]
    pdfs = sorted({r["file"] for r in rows})
    inputs = [tpath, LEVELING_PANEL, TOWNS_JSON] + [REPO / p for p in pdfs]
    built = {}
    for variant in ("preorigin", "postorigin"):
        pre = variant == "preorigin"

        def sel(df, col, conv=lambda x: x, pre=pre):
            if df.empty:
                return df
            v = df[col].map(conv)
            return df[v <= LAST_PRE_ROC + 1911] if pre else df[v > LAST_PRE_ROC + 1911]

        out = manifest.new_version_dir(PROD_LEVEL_PRE if pre else PROD_LEVEL_POST, root=root)
        tabs = {
            "county_intervals.csv": sel(iv, "survey_year"),
            "county_intervals_all_editions.csv": sel(iv_all, "survey_year"),
            "county_area_by_year.csv": sel(area, "year_label",
                                           lambda x: int(re.match(r"\d+", x)[0]) + 1911),
            "max_rate_by_region.csv": sel(mrate, "roc_year", lambda x: int(x) + 1911),
            "survey_dates.csv": sel(sdates, "edition_roc", lambda x: int(x) + 1911),
            "township_statements.csv": sel(st, "survey_year"),
            "township_status.csv": sel(ts, "survey_year"),
            "township_bounds.csv": sel(bd, "survey_year"),
        }
        if not pre:
            tabs["hsr_piers.csv"] = hsr
        for name, df in tabs.items():
            df = df.copy()
            df["flag"] = 0
            df["method"] = "wra_printed_value_double_transcribed"
            df["product_version"] = f"{variant}/{out.name}"
            _write(df, out / name)
        _write(unverified, out / "unverified.csv")
        v = {"schema": "x4-leveling-validation/1", "variant": variant,
             "transcription": n_val, "against_panel_2012_2021": val,
             "printed_inconsistencies": cons,
             "gate_rule": "all values double-transcribed and agreeing; printed county maximum "
                          "rates 2012-2021 within 1.0 cm/yr of the panel's screened maximum "
                          f"(benchmark-pair rates <= {SCREEN_CM_YR} cm/yr) in at least 60 % of "
                          "county-years (the panel is a subset of the same WRA network, so this "
                          "checks the transcription and the interval mapping, not the survey); "
                          "fixed before the comparison was run",
             }
        ms = val.get("max_rate_summary", {})
        gate = bool(ms) and ms["within_1.0"] >= 0.6 * ms["n"]
        v["gate_passed"] = gate
        (out / "validation.json").write_text(json.dumps(v, ensure_ascii=False, indent=1,
                                                        default=float), encoding="utf-8")
        if pre:
            kw = dict(fit_window_end="2022-06",
                      allowed_uses=["validator", "after_the_fact_analysis",
                                    "design_time_evidence"],
                      forbidden_uses=["scoring_target", "baseline", "datum_fit",
                                      "calibration_target", "prospective_scorer",
                                      "initial_conditions", "idw_field", "prior",
                                      "prior_width_input", "model_design", "selection",
                                      "forcing_frozen", "forcing_recorded_arm"])
        else:
            kw = dict(fit_window_end="2025-12",
                      allowed_uses=["validator", "after_the_fact_analysis"],
                      forbidden_uses=["scoring_target", "baseline", "datum_fit",
                                      "calibration_target", "prospective_scorer",
                                      "initial_conditions", "idw_field", "prior",
                                      "prior_width_input", "model_design", "selection",
                                      "forcing_frozen", "forcing_recorded_arm",
                                      "design_time_evidence", "statistical_layer_history"])
        m = manifest.build_manifest(
            out, product=PROD_LEVEL_PRE if pre else PROD_LEVEL_POST, layer="L1",
            inputs=inputs, code=code,
            parameters={"last_preorigin_survey_roc": LAST_PRE_ROC,
                        "models": {"pdf": "claude-opus-5-5", "text": "claude-sonnet-5-5"},
                        "acceptance": "pdf transcription equals the text (or second text) "
                                      "transcription field for field"},
            validation_report="validation.json",
            gate={"passed": gate, "criteria": v["gate_rule"]},
            personal_data="none (no names; station contacts never read)",
            extra={"wp": "X4", "rule_u7": (
                "post-origin observations: validator / after-the-fact analysis of the frozen "
                "and recorded-forcing runs only; never a gate, ladder decision, member choice, "
                "hyperparameter or pre-registered threshold") if not pre else (
                "leveling-derived aggregates up to May 2022; per plan 4.1 the 2012-2021 values "
                "come from the same leveling the column is calibrated on, so they validate the "
                "column only out of fold; the May 2022 survey is new"),
                "licence": "WRA website publications (政府網站資料); cite 經濟部水利署; kept in "
                           "gitignored results; HSR values are Taiwan High Speed Rail data as "
                           "printed by the WRA"},
            **kw)
        manifest.write_manifest(out, m)
        built[variant] = rel_to_repo(out)
        print(json.dumps({"variant": variant, "dir": rel_to_repo(out), "gate": gate,
                          "rows": {k: int(len(t)) for k, t in tabs.items()}},
                         ensure_ascii=False))
    print(json.dumps({"validation": {k: val.get(k) for k in ("max_rate_summary",
                                                            "towns_summary", "area")},
                      "inconsistencies": {k: len(v) for k, v in cons.items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
