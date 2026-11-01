"""Write N3SL.township_prospective_roc116-117.v1 (harness.prereg3, kind 'test'), rank 13.

N3S-L: a prospective test of the frozen page model (ZR), a statistical background layer (SL,
mean3_eb) and zero (Z) on WRA's printed leveling summaries for survey years ROC116 and
ROC117, read only through the new product version subsidence_public_reports/prospective
(user ruling 2026-11-01, fourth set: future survey intervals only, prospective_scorer
allowed; ROC115 excluded; SL's page role fixed now). L1 coast (C26, coastal C30), L2 county
area (C05), L3 HSR piers report-only, L4 dropped. Scored after N3 (rank 6), not before
2028-01-15.

Slots to fill before ``--write`` (each refused if missing or inconsistent):
  --freeze-dir     freeze_n3s_l.py freeze + verify output (ZR, SL, Z predictions, cells,
                   tolerance; verify.json identical; forward sha = the page model's; SL
                   source product granting statistical_layer_history AND baseline,
                   window_end <= 2021-12);
  --availability   the existence-only record of ``availability`` (WRA listing titles only, no
                   file fetched): ROC116 and ROC117 must be unlisted, checked <= 14 days before;
  --sl-role        report_only | eligible_ds (the user's choice; fixed now, never after
                   scoring);
  --expect-self-sha  this file's sha256.
``--write`` also refuses after 2026-11-30T16:00Z, on any blocking look or validation error.

Usage (repository root):
    python -I dataprep_staging/n3_township/prereg_n3s_l.py availability --out <json>
    python -I dataprep_staging/n3_township/prereg_n3s_l.py --validate [slots]
    python -I dataprep_staging/n3_township/prereg_n3s_l.py --write --freeze-dir D
        --availability F --sl-role R --expect-self-sha <sha>     (the lead only)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import freeze_n3s_l as F  # noqa: E402
import score_n3s_l as S  # noqa: E402
from common.paths import from_repo, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
PENDING = "PENDING"
FROZEN = tuple(f"{S.E}/{n}" for n in ("score_n3s_l.py", "freeze_n3s_l.py",
                                      "build_prospective_product.py", "prereg_n3s_l.py"))
CODE = ("dataprep_staging/x4_canal_leveling/lev_build.py",
        "dataprep_staging/x4_canal_leveling/lev_extract.py",
        "dataprep_staging/x4_canal_leveling/lev_pages.py",
        "dataprep_staging/x4_canal_leveling/config.py",
        "dataprep_staging/x5_nudge/prereg_x5nh.py")
CODE_AT_REGISTRATION = ("dataprep_staging/harness/prereg3.py",
                        "dataprep_staging/common/guard.py",
                        "dataprep_staging/common/manifest.py",
                        "dataprep_staging/common/uses.py",
                        "hydrophysics/twin/leveling.py")
SERIES = S.AVAIL_SERIES
AVAIL_MAX_AGE_DAYS = S.AVAIL_MAX_AGE_DAYS
PASSABILITY_N = (3, 4, 5, 6, 8)
PASSABILITY_SIGMA = (0.0, 0.5, 1.0, 1.5)
RULINGS = {
    "2026-11-01 fourth set, rank 13": "create subsidence_public_reports/prospective/v1 (future "
                                      "survey intervals only, prospective_scorer allowed; "
                                      "ROC115 excluded) and register a prospective "
                                      "township-summary test on ROC116/117; fix SL's page role "
                                      "now",
    "2026-11-01 fourth set, rank 6": "N3.piles_prospective approved; N3S-L's session order "
                                     "comes after N3",
    "2026-10-29 (1)": "claims ledger; narrow or withdraw unsupported page outputs (C05 area "
                      "tile withdrawn)",
    "2026-10-23": "retry rule: engineering retries free; scored tests run once",
}
ENGINEERING = ("a git-tracked module in code_at_registration_report_only may be restored to "
               "its registered sha256 (git checkout of that blob) before the scoring run, "
               "logged in the report; frozen_code and bound code are never edited")
SL_ROLE_TEXT = {
    "report_only": "SL's L1 and L2 readings are reported beside ZR under every outcome; no "
                   "claim, tile or page line comes from SL",
    "eligible_ds": "a coastal-background line C26b ('coastal sinking continued at its pre-2022 "
                   "background; statistical layer, not the physics column') is shown now as D "
                   "(registered, pending) in the evidence list only; on SL's combined L1 "
                   "'supported' it becomes S (prospective, ROC116/117 survey years only); on "
                   "'not_supported' it is withdrawn as decided (W); on 'undecided' the D line "
                   "is removed"}


Z_CLAUSE_READING = (
    "The plan says 'does at least as well as Z on both counts'. Read as two separate "
    "comparisons (lower met >= Z's AND upper broken <= Z's) it requires zero broken upper "
    "bounds, because Z (0 cm/yr) breaks an upper bound only if a printed county maximum is "
    "below -0.5 cm/yr; that cancels the plan's explicit 'at most 2 broken' allowance, so the "
    "two clauses of the same rule contradict each other. Registered reading: one comparison "
    "on both counts together, the net bound score (lower met - upper broken) >= Z's. The "
    "explicit numeric allowance (<= 2 broken) and lower-bound quota (ceil(0.5 x listed)) "
    "govern; the Z comparison stays well-defined for any competitor (with Z's usual 0/0 it "
    "does not bind beyond them). Chosen from the rule's logic alone, before any registered "
    "year exists; no 2023-25 listing was consulted for the choice. Alternatives considered: "
    "lower bounds only (equivalent here); drop the upper half of the Z clause (equivalent "
    "here); the literal reading (rejected: contradicts '<= 2', see the passability table)")


def passability_table() -> dict:
    """Synthetic design-time check (score_n3s_l.l1_passability; no data): P(supported in a
    year) for an arm equal to a synthetic truth field plus N(0, sigma) peak error, under the
    registered and the literal Z readings."""
    return {"what": "P(an arm is supported in one year | arm = synthetic truth field + "
                    "N(0, sigma) error on each township peak); generic truth ranges (listed "
                    "U(3, 6), unlisted U(0, 3) cm/yr), not taken from any WRA summary",
            "rows": [{"n_listed": n, "sigma_cm_yr": sg,
                      "net": round(S.l1_passability(n, sg, "net"), 3),
                      "literal": round(S.l1_passability(n, sg, "literal"), 3)}
                     for n in PASSABILITY_N for sg in PASSABILITY_SIGMA]}


def _sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _rec(p) -> dict:
    p = from_repo(p)
    return {"path": rel_to_repo(p), "sha256": S.sha256(p)}


def index_record(prereg_id_prefix: str, prereg_dir=None) -> dict | None:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rows = [r for r in S.index_rows(d) if str(r["id"]).startswith(prereg_id_prefix)]
    return rows[-1] if rows else None


# ----------------------------------------------------------------------------- availability
roc_status = S.roc_status                    # titles only (never the publication column)
availability_record = S.availability_record


def fetch_titles(series: str) -> list[dict]:
    """{'title', 'published'} of one WRA report series' listing pages (in memory; nothing
    saved). The ROC status is read from the title only; 'published' only dates the first
    listing (score_n3s_l.first_listed)."""
    from x4_canal_leveling import web
    from x4_canal_leveling.config import WRA_HOME, WRA_LIST
    from x4_canal_leveling.fetch_wra import parse_listing

    op = web.opener()
    web.get(op, WRA_HOME)
    url = WRA_LIST.format(series=series)
    st, _, body = web.get(op, url, referer=WRA_HOME)
    if st != 200:
        raise SystemExit(f"listing {series}: HTTP {st}")
    rows, pages = parse_listing(body.decode("utf-8", "replace"), series)
    for pg in [p for p in pages if p > 1]:
        st, _, b = web.get(op, f"{url}?Page={pg}", referer=url)
        rows += parse_listing(b.decode("utf-8", "replace"), series)[0]
    return [{"title": r["title"], "published": r["published_roc"]} for r in rows]


def availability_block(path) -> dict:
    if path is None:
        return {"status": PENDING}
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {"record": _rec(Path(path).resolve()), "checked": r.get("checked"),
           "series": r.get("series")}
    when = dt.datetime.fromisoformat(str(r.get("checked")).replace("Z", "+00:00"))
    age = (S._now() - when).days
    if r.get("id") != S.ID or r.get("kind") != "availability_check":
        out["status"] = "not this registration's record"
    elif set(r.get("series") or {}) != set(SERIES):
        out["status"] = f"series missing: needs {SERIES}"
    elif not r.get("unpublished_116_117"):
        out["status"] = "ROC116 or ROC117 is listed: do not register"
    elif age > AVAIL_MAX_AGE_DAYS or age < 0:
        out["status"] = f"record is {age} days old (max {AVAIL_MAX_AGE_DAYS})"
    else:
        out["status"] = "ok"
    return out


# ----------------------------------------------------------------------------- freeze slot
def freeze_block(fdir) -> dict:
    """The frozen predictions: every file hashed; 'ok' only with an identical rebuild, the
    page model's forward and an SL source granting both SL uses (statistical_layer_history,
    baseline; neither forbidden) to 2021-12 with its gate passed."""
    if fdir is None:
        return {"status": PENDING, "dir": None, "files": {}}
    d = Path(fdir)
    out = {"dir": rel_to_repo(d), "files": {}}
    try:
        fz = S.load_freeze(d)
    except S.EngineeringFailure as e:
        return {**out, "status": f"invalid: {e}"}
    s = fz["summary"]
    out["files"] = {n: _rec(d / n) for n in [*s["files"], "summary.json"]}
    out["sl_params"] = s.get("sl")
    out["provenance"] = s.get("provenance")
    v = json.loads((d / "verify.json").read_text()) if (d / "verify.json").is_file() else {}
    csv = {n: h for n, h in s["files"].items() if n.endswith(".csv")}
    prov = s.get("provenance") or {}
    slm = prov.get("sl_product_manifest") or {}
    sl_ok = False
    slp = from_repo(slm["path"]) if slm.get("path") else None
    if slp is not None and slp.is_dir():     # freeze records the product directory
        slp = slp / "MANIFEST.json"
    if slp is not None and slp.is_file():
        m = json.loads(slp.read_text(encoding="utf-8"))
        sl_ok = (S.sha256(slp) == slm.get("sha256")
                 and all(u in m.get("allowed_uses", []) for u in F.SL_USES)
                 and not any(u in m.get("forbidden_uses", []) for u in F.SL_USES)
                 and str(m.get("fit_window_end", "9999-12")) <= "2021-12"
                 and bool((m.get("gate") or {}).get("passed")))
        out["sl_product"] = {"product": m.get("product"), "version": m.get("version"),
                             "fit_window_end": m.get("fit_window_end"),
                             "allowed_uses": m.get("allowed_uses")}
    if v.get("identical") is not True or v.get("files") != csv or \
            v.get("summary_sha256") != S.sha256(d / "summary.json"):
        out["status"] = "not verified"
    elif (prov.get("forward") or {}).get("sha256") != S.MODEL_SHA:
        out["status"] = "forward is not the page model's"
    elif not sl_ok:
        out["status"] = ("SL source does not grant statistical_layer_history and baseline "
                         "to 2021-12")
    elif any(fz["tol"].get(c) is None for c in S.COUNTIES):
        out["status"] = "no pre-origin area tolerance for a county"
    else:
        out["status"] = "ok"
    out["verify"] = {k: v.get(k) for k in ("identical", "problems")}
    return out


# ----------------------------------------------------------------------------- the document
def unit(roc: int) -> dict:
    ids = [f"roc{roc}:{S.COAST_COUNTY[t]}:{t}" for t in S.COAST] + \
        [f"roc{roc}:{c}" for c in S.COUNTIES]
    return {"description": f"WRA printed leveling summary for survey year ROC{roc} "
                           f"({S.NOMINAL[roc][0]}..{S.NOMINAL[roc][1]}): the 11 coastal "
                           "townships' listing and bounds and the two counties' "
                           "significant-subsidence area, read only from "
                           f"{S.PRODUCT}; no well, benchmark or model cell",
            "ids": ids, "window": S.DATA_SCOPE[roc].split(":")[1]}


def tests_for(roc: int, sl_role: str) -> list[dict]:
    common = {"data_scope": S.DATA_SCOPE[roc], "fresh_units": [S.UNITS[roc]],
              "external_units": {S.UNITS[roc]: unit(roc)},
              "label": "prospective: the survey year's summary did not exist at registration"}
    l1 = {
        "test_id": S.TESTS[roc]["L1"], **common,
        "claim": "C26 and the coastal part of C30 (ZR); C26b if SL's role is eligible_ds",
        "what": f"Do the frozen arms' coastal township peak rates meet WRA's printed ROC{roc} "
                "township bounds?",
        "metric": {
            "townships": f"the 11 coastal townships {list(S.COAST)} (X5NH: >= 50 % of their "
                         f"twin cells west of {S.DISTAL_KM:g} km), atlas 2021 polygons; a "
                         "cell belongs to a township when its centroid lies inside",
            "bounds": "the product's township_bounds rows (lev_build.bounds): lower bound "
                      f"{S.LISTED_FLOOR:g} cm/yr when listed by every source, upper bound the "
                      "printed bound; an ambiguous listing (sources disagree) carries neither; "
                      f"tolerance {S.TOL:g} cm/yr on every bound (X5NH's)",
            "ZR": "max over the township's cells of the frozen forward's baseline ensemble-mean "
                  "rate over the PRINTED survey interval (county_intervals start_ym..end_ym; "
                  "nominal May-to-May if not printed), recomputed from the hashed forward; the "
                  "nominal-interval values frozen now must be reproduced exactly first",
            "SL": "aggregated like ZR: max over the township's cells of SL's frozen cell "
                  "field (each cell takes its nearest benchmark's mean3_eb rate within "
                  f"{F.SL_RADIUS_KM:g} km; persistence: interval-independent); no cell with a "
                  "value -> no prediction (lower bound missed, upper bound kept). The max over "
                  "single benchmarks inside the township is report-only (lev_build.bounds: "
                  "unlisted upper bounds apply to the contoured field, not to benchmarks)",
            "Z": "0 cm/yr everywhere",
            "report_only": ["ZR's township mean and distal-cell peak",
                            "SL's benchmark peak", "per-township rows of every arm"]},
        "informative_gate": f">= {S.L1_MIN_LISTED} coastal townships listed (lower bounds) in "
                            "the survey year; evaluable only if Yunlin's township list for "
                            "the year is verified",
        "decision_rule": {
            "per_year_per_arm": f"'supported' iff lower bounds met >= ceil({S.L1_LOWER_FRAC} x "
                                f"listed) AND upper bounds broken <= {S.L1_MAX_UPPER_BROKEN} "
                                "AND (lower met - upper broken) >= Z's (lower met - upper "
                                "broken); else 'not_supported'; 'not_informative' below the "
                                "gate; 'not_evaluable' without a verified Yunlin list",
            "z_clause_reading": Z_CLAUSE_READING,
            "passability": passability_table(),
            "combined_over_years": "per arm: 'supported' iff supported in every decided year "
                                   "(ROC116, ROC117) and at least one decided; 'undecided' if "
                                   "none decided; otherwise 'not_supported'",
            "no_selection": "the verdict is reported per arm and never selects between arms",
            "sl_role": f"{sl_role}: {SL_ROLE_TEXT.get(sl_role, PENDING)}"}}
    l2 = {
        "test_id": S.TESTS[roc]["L2"], **common,
        "claim": "C05 (area tile, county area sinking > 3 cm/yr)",
        "what": f"Is the frozen model's ROC{roc} county area sinking > {S.AREA_RATE:g} cm/yr "
                "close to WRA's printed significant-subsidence area, and closer than SL's?",
        "metric": {
            "printed": "first verified source of: table 1-1 (county_area_by_year, year label "
                       f"{roc}), table 2-x (county_intervals area_gt3_km2), the statement's "
                       "significant area; disagreements reported",
            "ZR": f"the page's area definition on the survey interval: active cells (1 km2) "
                  f"of the county whose rate exceeds {S.AREA_RATE:g} cm/yr, counted on the "
                  "ensemble-mean field",
            "SL": "the same count on SL's cell field (each cell takes its nearest benchmark's "
                  f"rate within {F.SL_RADIUS_KM:g} km; none -> not counted)",
            "tolerance": "per county, median |change| of the printed significant area between "
                         f"consecutive pre-origin survey years {F.TOL_YEARS[0]}-"
                         f"{F.TOL_YEARS[1]} (preorigin v1, design_time_evidence), frozen now",
            "tolerance_role": "a test-design input of this registration only (the zero-area "
                              "branch of L2); never a model gate, ladder decision, member "
                              "choice, hyperparameter or threshold of any model (not the "
                              "'selection' use, which preorigin v1 forbids)",
            "L3_report_only": "HSR piers: Spearman over printed pier rates and the model at "
                              "the pier cells, and max within x1.5; needs a pier location "
                              "table (metadata only) frozen with the predictions, else 'not "
                              "evaluable'; never gates (C06 unchanged)"},
        "decision_rule": {
            "county_year": f"printed > 0: model/printed in [1/{S.AREA_RATIO:g}, "
                           f"{S.AREA_RATIO:g}] AND |log(model/printed)| < "
                           "|log(SL/printed)| (SL = 0 counts as infinitely far); printed = "
                           "0: model <= tolerance AND model < SL (a tie fails)",
            "combined": f"over the 4 county-years (Yunlin, Changhua x ROC116, ROC117): "
                        f"'supported' iff >= {S.L2_MIN_PASS} pass; 'undecided' if fewer than "
                        f"{S.L2_MIN_PASS} are evaluable; else 'not_supported'",
            "not_evaluable": "a county-year without a verified printed area (or a zero "
                             "printed area without a tolerance)"},
        "informative_gate": f">= {S.L2_MIN_PASS} evaluable county-years over both years"}
    return [l1, l2]


def _manifest_path(path: str) -> str:
    """A product directory in the freeze provenance stands for its MANIFEST.json."""
    return f"{path}/MANIFEST.json" if from_repo(path).is_dir() else path


def doc(freeze_dir=None, availability=None, sl_role: str | None = None,
        prereg_dir=None) -> dict:
    role = sl_role or "report_only"
    fz = freeze_block(freeze_dir)
    av = availability_block(availability)
    n3 = index_record(S.N3_PREFIX, prereg_dir)
    frozen = {p: S.sha256(REPO / p) for p in FROZEN if (REPO / p).is_file()}
    code = {p: S.sha256(REPO / p) for p in CODE}
    code_reg = {p: S.sha256(REPO / p) for p in CODE_AT_REGISTRATION}
    fwd = REPO / S.FORWARD
    inputs = sorted(set(frozen) | set(code) | {r["path"] for r in fz["files"].values()}
                    | ({av["record"]["path"]} if "record" in av else set())
                    | ({S.FORWARD} if fwd.is_file() else set())
                    | {_manifest_path(p["path"]) for k, p in (fz.get("provenance") or {}).items()
                       if k in ("sl_product_manifest", "preorigin_manifest", "towns")})
    tests = tests_for(116, role) + tests_for(117, role)
    order = [
        "(1) the 2028 session of N2C, N2, E02/E02c, N3F (their registered order) is unchanged",
        "(2) N3 (rank 6, the registered N3.piles_prospective* id), after N2 and N2C step (1); "
        "its 溪州 report-only arm after N2C",
        f"(3) {S.ID} ROC116: availability shows the ROC116 summary published; N3 scored "
        "(score look + report) or a lead-written not_scorable record naming it, or no N3 "
        "registration exists; open-session logs both ROC116 score looks; only then the "
        "reports are fetched, transcribed (lev_extract) and the prospective product built; "
        "score",
        f"(4) {S.ID} ROC117: the same, combining with ROC116's report (or its not_published "
        "record); if ROC117 is not listed by its deadline, the not-published record for ROC117 "
        "carries the final combination of ROC116 alone",
        "(1)-(3) are enforced by open-session: a fresh availability record (<= 14 days) listing "
        "the year's summary or briefing (007005/007007), first listed on or before the year's "
        "no_later_than date (the deadline applies to the first listing, not the session date)"]
    disclosures = {
        "looked_2023_25": "the 2023-25 (ROC112-114) coastal listings (6, 5 and 0 coastal "
                          "townships listed; logged 2026-10-28, X5NH validator readings) were "
                          "seen before these rules were written. The 0.5 cm/yr tolerance and "
                          "the coastal list are X5NH's; the 1.5 band is N1's; ceil(0.5 x "
                          "listed), <= 2 broken and 3 of 4 county-years are copied unchanged "
                          "from the round-5 plan. Copying does not undo the look (those plans "
                          "were written after it); no 2023-25 value set a threshold here",
        "physics_near_z": "the page column barely moves on the coast (the distal zone does "
                          "not sink in the model), so ZR is near Z there and L1 is in effect a "
                          "test of SL against Z; ZR is expected to fail L1 whenever it is "
                          "informative",
        "roc116_starts_before_registration": "ROC116's interval starts 2026-05, before "
                                             "registration; heads to 2026-09 are held but "
                                             "neither ZR (frozen at the 2022-12 origin) nor "
                                             "SL (leveling to 2021-09) uses them",
        "sl_not_independent": "SL is built from the column's calibration leveling (pre-origin), "
                              "through a product granting statistical_layer_history (SL's "
                              "history, L1) and baseline (SL as the L2 comparator), "
                              "fit_window_end 2021-12; rows flag 0 (observed), the 50 cm/yr "
                              "datum-artefact screen applied per interval downstream",
        "area_definition": "WRA interpolates leveling into a contoured area; the model counts "
                           "1 km2 cells; the x1.5 band absorbs part of this, not a "
                           "definitional bias",
        "publication_lag": "ROC116 may first appear in the June/July 2027 monthly reports "
                           "(007001), the summary later; the first verdict comes after N3's",
        "monthly_report_channel": "the June/July monthly reports of 2027 and 2028 state the "
                                  "year's leveling result; any extraction of them before this "
                                  "test's session (N3's pile months included) transcribes "
                                  "only pile tables, never the statement pages",
        "roc115": "ROC115 (2025-05..2026-05) is excluded: its interval overlaps looked InSAR and "
                  "pile months; the builder drops its rows; it stays unread until ROC116 is "
                  "scored and can never set or change a threshold here",
        "multiplicity": "the coastal summaries were read as validator in X5NH (2023-25); this "
                        "registration adds 2 tests per survey year; N3 owns the pile months; "
                        "each test decides only its own claim, no family-wise claim",
        "l4_dropped": "the response test (L4) is dropped: about 11 townships and class-level "
                      "outcomes; expected not informative",
        "z_clause": "see L1 z_clause_reading (changed from the plan's literal two-count "
                    "reading before registration, on the rule's logic; the passability table "
                    "is synthetic)"}
    page_mapping = {
        "before_scoring": {
            "C05": "W, with the D sub-label 'decided by N3S-L L2 on WRA ROC116/117 "
                   "significant-subsidence area (not before 2028-01-15)'",
            "C26": "N, with the D sub-label 'coastal status quo decided by N3S-L L1 (expected: "
                   "the physics column fails; a statistical background layer is the "
                   "competitor)'",
            "C30_coastal": "N, with the D sub-label 'coastal part decided by N3S-L L1'",
            "C29": "L unchanged; adds 'coastal cells: see N3S-L'",
            "C26b": "shown as D in the evidence list" if role == "eligible_ds" else
                    "not shown (SL report-only)",
            "unchanged": "C06 (L3 report-only), C25 (two survey years cannot decide the long "
                         "run), C27 (L4 dropped), C31 (rank 6's pile claim)"},
        "after_scoring": "score_n3s_l.final_mapping (frozen code): C26 / C30 coastal from ZR's "
                         "combined L1 ('supported' stays N with a decided label, no S; "
                         "'not_supported' -> 'coastal forecasts failed a registered "
                         "prospective test'; 'undecided'); C05 from L2 ('supported' -> "
                         "returns as a narrowed S: the county area sinking > 3 cm/yr for "
                         "ROC116/117 only; 'not_supported' -> W decided; 'undecided' -> W); "
                         f"C26b by SL's fixed role ({role}). If ROC117 is never published "
                         "the same mapping is applied to ROC116 alone (not-published "
                         "--survey-roc 117 --stage1-report)",
        "never": "no cut number, no avoided-subsidence size and no policy difference becomes "
                 "S; no Changhua subsidence lever; ROC115 never scored; SL's role is never "
                 "changed after any look"}
    d = {
        "id": S.ID, "wp": S.WP, "kind": "test",
        "title": "N3S-L: frozen page model, statistical layer and zero against WRA's printed "
                 "ROC116/117 coastal township bounds and county significant-subsidence area",
        "why": "The township summaries were validator-only (both v1 versions forbid "
               "prospective_scorer). User ruling 2026-11-01 (fourth set) creates a prospective "
               "version restricted to survey intervals ending after registration, so the "
               "coastal and area claims can be decided by a registered test instead of left "
               "untested.",
        "tests": tests,
        "product_definition": {
            "product": S.PRODUCT, "first_version": "v1 = the ROC116 stage; the ROC117 stage "
                                                   "writes v2 with the same frozen builder "
                                                   "(user confirmation of v2 recorded in "
                                                   "rulings before --write)",
            "builder": f"{S.E}/build_prospective_product.py",
            "allowed_uses": ["prospective_scorer", "validator", "after_the_fact_analysis"],
            "scope": f"survey years {list(S.SURVEYS)} only; ROC{S.EXCLUDED_ROC_MAX} and earlier "
                     f"and intervals ending on or before {S.REG_MONTH} dropped",
            "acceptance": "double transcription agrees field for field (the v1 rule); "
                          "unverified fields are never used",
            "existing_versions": "preorigin/v1 and postorigin/v1 stay validator-only"},
        "sl_page_role": {"role": role if sl_role else PENDING, "text": SL_ROLE_TEXT.get(role),
                         "fixed": "at registration (user ruling 2026-11-01: fix SL's page role "
                                  "now); never decided after scoring"},
        "binds": {"forward": {"path": S.FORWARD, "sha256": S.MODEL_SHA},
                  "frozen_predictions": fz, "code": code,
                  "code_at_registration_report_only": code_reg, "engineering": ENGINEERING,
                  "n3_registration": n3 or "not in the index at drafting: if registered later "
                                           "N3S-L is still scored after it (scorer order "
                                           "check)"},
        "availability": av,
        "frozen_code": frozen,
        "scoring": {
            "not_before": "2028-01-15",
            "no_later_than": dict((str(k), v) for k, v in S.NO_LATER_THAN.items()),
            "order": order,
            "scorer": f"{S.E}/score_n3s_l.py open-session / score / not-published",
            "precondition": "the product MANIFEST lists prospective_scorer, its gate passed and "
                            "it names this registration; checked before any table is read",
            "engineering": ENGINEERING},
        "sealed_until_scoring": "the ROC115, ROC116 and ROC117 WRA summaries and briefings, "
                                "and the statement pages of the June/July 2027 and 2028 "
                                "monthly reports; nothing after 2021-09 enters SL and nothing "
                                "after 2022-12 enters ZR",
        "page_mapping": page_mapping,
        "disclosures": disclosures,
        "attempt_policy": "User rule 2026-10-23: engineering failures (crash, bug, missing "
                          "file, a transcription that fails the acceptance rule) may be fixed "
                          "and re-run before the scoring look, each logged; after the score "
                          "look an engineering failure writes no verdict and may be fixed and "
                          "re-run, every run listed; each test is scored ONCE; failed, "
                          "undecided or not-evaluable results are reported as such and never "
                          "re-scored on these survey years; after a failure at most ONE "
                          "follow-up registration (new ids, 'post-failure, exploratory', never "
                          "validation) which cannot reuse ROC116/117; the final report lists "
                          "every attempt",
        "rulings": RULINGS,
        "zones": NA("no fit; the frozen forward run's own zones"),
        "priors": disclosures,
        "burn_in": NA("no fit"), "folds": NA("prospective survey years"),
        "fresh_units": [S.UNITS[r] for r in S.SURVEYS],
        "thresholds": {"listed_floor_cm_yr": S.LISTED_FLOOR, "tol_cm_yr": S.TOL,
                       "l1_min_listed": S.L1_MIN_LISTED, "l1_lower_frac": S.L1_LOWER_FRAC,
                       "l1_max_upper_broken": S.L1_MAX_UPPER_BROKEN,
                       "area_rate_cm_yr": S.AREA_RATE, "area_ratio": S.AREA_RATIO,
                       "l2_min_pass": S.L2_MIN_PASS, "sl_radius_km": F.SL_RADIUS_KM,
                       "sl_n": F.SL_N, "sl_min_n": F.SL_MIN_N, "sl_last_end": F.SL_LAST_END,
                       "sl_max_rate_cm_yr": F.SL_MAX_RATE, "tol_years": list(F.TOL_YEARS)},
        "gate_scores": NA("no weighted gate; each decision rule is reported as it falls"),
        "primary_model": {"name": "datum_gate_innov (the page model's frozen forward run)",
                          "forward_sha256": S.MODEL_SHA,
                          "competitors": "SL mean3_eb (frozen), Z (zero)"},
        "inputs": inputs,
        "interpretation": "Whether the frozen page model, a statistical background layer and "
                          "zero match WRA's next two printed leveling summaries on the coast "
                          "and in county area. A pass can at most narrow the area claim for "
                          "those survey years; nothing about cut sizes, policy responses or "
                          "the long run.",
        "written_by": f"{S.E}/prereg_n3s_l.py"}
    return d


# ----------------------------------------------------------------------------- CLI
def problems_for_write(d: dict, sl_role, expect_self_sha) -> list[str]:
    bad = []
    if d["binds"]["frozen_predictions"]["status"] != "ok":
        bad.append(f"freeze slot: {d['binds']['frozen_predictions']['status']}")
    if d["availability"]["status"] != "ok":
        bad.append(f"availability slot: {d['availability']['status']}")
    if sl_role not in S.SL_ROLES:
        bad.append(f"--sl-role must be one of {S.SL_ROLES} (the user's choice, fixed now)")
    if set(d["frozen_code"]) != set(FROZEN):
        bad.append(f"frozen code incomplete: {sorted(set(FROZEN) - set(d['frozen_code']))}")
    if not (REPO / S.FORWARD).is_file() or S.sha256(REPO / S.FORWARD) != S.MODEL_SHA:
        bad.append("forward npz missing or changed")
    self_sha = S.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != this script {self_sha}")
    if S._now() >= S.REG_DEADLINE:
        bad.append(f"DEADLINE PASSED ({S.REG_DEADLINE:%Y-%m-%dT%H:%MZ})")
    return bad


def check_blocking(d: dict, prereg_dir=None) -> list:
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    idx = prereg3.membership_index(pdir)
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            if not idx.known(u):
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
    return prereg3.blocking_looks(d["tests"], None, pdir, idx, test_time=S._now())


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "availability":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv)
        rec = availability_record({s: fetch_titles(s) for s in SERIES})
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise SystemExit(f"{out} exists")
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"unpublished_116_117": rec["unpublished_116_117"],
                          "series": rec["series"]}, ensure_ascii=False))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="write the registration (the lead)")
    ap.add_argument("--freeze-dir", type=Path, default=None)
    ap.add_argument("--availability", type=Path, default=None)
    ap.add_argument("--sl-role", choices=S.SL_ROLES, default=None)
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.freeze_dir, a.availability, a.sl_role, prereg_dir=prereg_dir)
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    block = check_blocking(d, prereg_dir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", _sha_text(d))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("freeze:", d["binds"]["frozen_predictions"]["status"], "| availability:",
          d["availability"]["status"], "| sl role:", d["sl_page_role"]["role"])
    print("this script sha256:", S.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = problems_for_write(d, a.sl_role, a.expect_self_sha)
    if errs or block or bad:
        print("REFUSED:", errs + [str(b) for b in block] + bad)
        return 1
    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
