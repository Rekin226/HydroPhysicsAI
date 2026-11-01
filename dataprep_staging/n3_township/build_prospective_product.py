"""Build ``subsidence_public_reports/prospective/v<N>`` (user ruling 2026-11-01, fourth set).

The prospective version of the WRA leveling summaries, the scorer of N3S-L
(N3SL.township_prospective_roc116-117.v1). It holds ONLY survey years whose interval ends
after the registration month (ROC116 = 2026-05..2027-05 and ROC117 = 2027-05..2028-05); every
row of ROC115 or earlier, and every row of a year not registered, is dropped and counted
(never written). Its MANIFEST lists ``prospective_scorer`` (plus ``validator`` and
``after_the_fact_analysis`` for use after scoring) and forbids selection, design,
calibration, baselines, priors, design-time evidence and statistical-layer history; the
pre- and post-origin versions stay validator-only.

The tables are those of the v1 versions, built by the same code
(``x4_canal_leveling.lev_build``: interval_table, statements, township_status, bounds,
simple_table) from a ``lev_extract`` transcription (two independent reads; a field is used only
when they agree, the v1 acceptance rule). Validation against the leveling panel does not apply
(the panel ends 2021).

Order (score_n3s_l.py): this builder refuses a survey year unless the registration is in the
index and both score looks of that year are logged (``open-session``), so no value of a
registered year is read into a product before the scoring look.

    python -I dataprep_staging/n3_township/build_prospective_product.py \
        <transcriptions.json> --survey-roc 116 [--survey-roc 117]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_n3s_l as S  # noqa: E402

ALLOWED = ["prospective_scorer", "validator", "after_the_fact_analysis"]
FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "calibration_target",
             "initial_conditions", "idw_field", "prior", "prior_width_input", "model_design",
             "selection", "forcing_frozen", "forcing_recorded_arm", "design_time_evidence",
             "statistical_layer_history"]
GATE_RULE = ("every value used is double-transcribed and agreeing (the v1 acceptance rule); "
             "no row of a survey year <= ROC115, of a year not registered, or of an interval "
             "ending on or before the registration month; each included survey year carries a "
             "verified Yunlin township list or is reported incomplete; fixed 2026-11 before "
             "any registered year existed")
RULING = ("user ruling 2026-11-01 (fourth set), rank 13: create subsidence_public_reports/"
          "prospective/v1 (future survey intervals only, prospective_scorer allowed; ROC115 "
          "excluded) and register a prospective township-summary test on ROC116/117")


def registration(prereg_dir) -> dict:
    rows = [r for r in S.index_rows(prereg_dir) if r["id"] == S.ID]
    if not rows:
        raise SystemExit(f"{S.ID} is not in the registry index: nothing is built before it")
    return rows[-1]


def _year_of(df: pd.DataFrame, col: str, conv=lambda x: x) -> pd.Series:
    if df.empty or col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col].map(conv), errors="coerce")


def select_years(tabs: dict, rocs: tuple[int, ...]) -> tuple[dict, dict]:
    """Keep rows of the registered survey years whose interval ends after REG_MONTH; count
    the dropped rows per reason (no value is kept or printed)."""
    keep, dropped = {}, {}
    years = {r + 1911 for r in rocs}
    conv = {"county_area_by_year.csv": ("year_label", lambda x: int(
                str(x).strip().split("(")[0][:3]) + 1911 if str(x).strip()[:3].isdigit()
                else None),
            "max_rate_by_region.csv": ("roc_year", lambda x: int(x) + 1911),
            "survey_dates.csv": ("edition_roc", lambda x: int(x) + 1911),
            "hsr_piers.csv": ("roc_year", lambda x: int(x) + 1911)}
    for name, df in tabs.items():
        col, f = conv.get(name, ("survey_year", lambda x: x))
        y = _year_of(df, col, f)
        ok = y.isin(sorted(years)) if len(df) else pd.Series(dtype=bool)
        if "end_ym" in df.columns and len(df):
            ok &= df["end_ym"].astype(str) > S.REG_MONTH
        dropped[name] = {"le_roc115": int((y <= S.EXCLUDED_ROC_MAX + 1911).sum()),
                         "other_years_or_unparsed": int((~ok).sum()
                                                        - (y <= S.EXCLUDED_ROC_MAX + 1911).sum())}
        keep[name] = df[ok.to_numpy()] if len(df) else df
    return keep, dropped


def completeness(keep: dict, rocs) -> dict:
    st = keep.get("township_status.csv", pd.DataFrame())
    out = {}

    def has(c: str, y: int) -> bool:
        return bool(len(st)) and bool(((st["county"] == c) & (st["survey_year"] == y)).any())

    for r in rocs:
        y = r + 1911
        out[str(r)] = {"yunlin_list": has("雲林", y), "changhua_list": has("彰化", y),
                       "rows": {n: int((_year_of(df, "survey_year") == y).sum())
                                for n, df in keep.items() if "survey_year" in df}}
    return out


def build(transcriptions, rocs, *, root=None, prereg_dir=None, towns=None,
          require_session=True) -> Path:
    from common import manifest
    from common.paths import DATAPREP, PREREG_DIR
    from x4_canal_leveling import lev_build as LB

    rocs = tuple(sorted(set(int(r) for r in rocs)))
    bad = [r for r in rocs if r not in S.SURVEYS]
    if bad or not rocs:
        raise SystemExit(f"only registered survey years {S.SURVEYS} may be built (got {rocs})")
    pdir = Path(prereg_dir) if prereg_dir else PREREG_DIR
    reg = registration(pdir)
    if require_session:
        for r in rocs:
            if not S.session_status(pdir, r)["ok"]:
                raise SystemExit(f"ROC{r}: the score looks are not logged; run score_n3s_l.py "
                                 "open-session first")
    tpath = Path(transcriptions)
    rows = LB.load(tpath)
    towns_from_atlas = towns is None
    towns = towns or LB.town_lists()
    iv, iv_all = LB.interval_table(rows, towns) if any(
        r["kind"] in ("interval_chc", "interval_yl") for r in rows) else (pd.DataFrame(),
                                                                          pd.DataFrame())
    st = LB.statements(rows, towns)
    ts = LB.township_status(st, iv if len(iv) else _empty_iv(), towns) if len(st) else \
        pd.DataFrame()
    tabs = {"county_intervals.csv": iv, "county_intervals_all_editions.csv": iv_all,
            "county_area_by_year.csv": LB.simple_table(rows, "area_county",
                                                       ["county", "year_label"]),
            "max_rate_by_region.csv": LB.simple_table(rows, "max_rate", ["region", "roc_year"]),
            "survey_dates.csv": LB.simple_table(rows, "survey_dates", ["region"]),
            "township_statements.csv": st, "township_status.csv": ts,
            "township_bounds.csv": LB.bounds(ts) if len(ts) else pd.DataFrame(),
            "hsr_piers.csv": LB.simple_table(rows, "hsr", ["chainage", "roc_year"])}
    keep, dropped = select_years(tabs, rocs)
    if not any(len(df) for df in keep.values()):
        raise SystemExit("no row of a registered future survey year: nothing to build")
    comp = completeness(keep, rocs)
    out = manifest.new_version_dir(S.PRODUCT, root=root or DATAPREP)
    for name, df in keep.items():
        df = df.copy()
        df["flag"] = 0
        df["method"] = "wra_printed_value_double_transcribed"
        df["product_version"] = f"prospective/{out.name}"
        df.to_csv(out / name, index=False)
    n_val = {"rows_total": len(rows), "rows_verified": int(sum(r["verified"] for r in rows)),
             "rows_unverified": int(sum(not r["any_verified"] for r in rows))}
    gate = not any(_year_of(df, "survey_year").le(S.EXCLUDED_ROC_MAX + 1911).any()
                   for df in keep.values() if "survey_year" in df)
    v = {"schema": "n3sl-prospective-validation/1", "survey_rocs": list(rocs),
         "transcription": n_val, "completeness": comp, "dropped_rows": dropped,
         "gate_rule": GATE_RULE, "gate_passed": bool(gate)}
    (out / "validation.json").write_text(json.dumps(v, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    pdfs = sorted({r["file"] for r in rows})
    inputs = [tpath] + [Path(p) if Path(p).is_absolute() else REPO / p for p in pdfs]
    if towns_from_atlas:
        inputs.append(LB.TOWNS_JSON)
    end = max([str(x) for x in keep["county_intervals.csv"].get("end_ym", [])] +
              [S.NOMINAL[r][1] for r in rocs])
    m = manifest.build_manifest(
        out, product=S.PRODUCT, layer="L1", inputs=inputs,
        code=[Path(__file__), REPO / "dataprep_staging/x4_canal_leveling/lev_build.py",
              REPO / "dataprep_staging/x4_canal_leveling/lev_extract.py",
              REPO / "dataprep_staging/x4_canal_leveling/config.py"],
        parameters={"registration_id": S.ID, "registration_sha256": reg["sha256"],
                    "registration_created": reg["created"], "survey_rocs": list(rocs),
                    "excluded": f"ROC{S.EXCLUDED_ROC_MAX} and earlier; intervals ending on or "
                                f"before {S.REG_MONTH}",
                    "acceptance": "pdf transcription equals the text (or second text) "
                                  "transcription field for field"},
        fit_window_end=end, validation_report="validation.json",
        gate={"passed": bool(gate), "criteria": GATE_RULE},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none (no names; station contacts never read)",
        extra={"wp": S.WP, "ruling": RULING,
               "rule_u7": "future survey intervals only (ending after the registration "
                          "month); scorer of the registered N3S-L tests; never a gate, "
                          "ladder decision, member choice, hyperparameter, calibration "
                          "target or threshold of any model",
               "licence": "WRA website publications (政府網站資料); cite 經濟部水利署; kept "
                          "in gitignored results"})
    manifest.write_manifest(out, m)
    return out


def _empty_iv() -> pd.DataFrame:
    return pd.DataFrame(columns=["county", "survey_year", "max_rate_cm_yr", "max_rate_township",
                                 "area_gt3_km2"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("transcriptions")
    ap.add_argument("--survey-roc", type=int, action="append", required=True)
    a = ap.parse_args(argv)
    out = build(a.transcriptions, a.survey_roc)
    m = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    print(json.dumps({"dir": str(out), "gate": m["gate"]["passed"],
                      "rows": {o["path"]: o.get("rows") for o in m["outputs"]}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
