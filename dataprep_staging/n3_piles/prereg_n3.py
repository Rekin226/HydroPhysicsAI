"""Write N3.piles_prospective.v1 (harness.prereg3, kind 'test') before 2026-11-30T16:00Z.

The one pile registration of round 5 (judge rank 6; user rulings 2026-11-01, fourth set):
the WRA printed deep-pile months 2026-12..2027-11 (24-month continuation to 2028-11) at the
two Yunlin piles anchored at 300 m or deeper (土庫國中 400 m, 300 m sensor as the declared
fallback; 中科虎尾 300 m), and a report-only Changhua arm at 溪州國小 300 m.

* T-conversion (claim C31): compaction per metre of head change, rho = g_obs / g_mod, N2's
  ratio rule; its ROLE (gating at 12 or 24 months, or report-only) is fixed by G0 before
  registration (g0_n3.py result, hashed here).
* T-level (claims C09/C30, narrowed to the 2 pile cells): the frozen ZR column against zero
  and the pre-origin leveling-cell rate (export_leveling_competitor.py product, hashed).
* T-溪州 (C13c/C27 wording only): season-aggregate elastic coefficient, report-only.

Slots that must be filled before ``--write``:
  --mapping-dir       build_mapping.py output (mapping.json, model_side.json);
  --g0-result         g0_n3.py result.json run on that model_side.json with this code;
  --leveling-product  the n3_leveling_cell_rate product directory (MANIFEST grants baseline).
If G0 fails at both horizons, N3 is NOT registered (the judge: 'else N3 is not registered
as gating'; the 2026-11-01 ruling approved N3 with 'G0 first, per the judge'). Registering
T-level as the only gating test with the conversion lines report-only reverses that literal
rule, so ``--g0-fail-mode level-only`` also needs ``--ruling <results/dataprep/rulings/*.json>
--ruling-id <id>``: a user ruling record (schema dataprep-rulings/1, ruling_by 'user') whose
ruling names this registration in ``applies_to``; its question and decision are copied into
the document. Without that ruling the ledger keeps C31 'untested: G0 failed', C27 'untested
(no registered test)' and no N3 sub-label on C09/C30 ('default_without_ruling' in the page
mapping). ``--write`` also refuses after the deadline, on a blocking look, on a pile look
whose scope reaches 2026-02 or later, or if --expect-self-sha is not this file's sha.

Every G0 attempt is disclosed: the archived runs under <g0 dir>/../archive (run_n3.sh
archive) with their sha, reason and outcome, plus the builder's scratch run (SCRATCH_G0).

Usage (repository root):
    python -I dataprep_staging/n3_piles/prereg_n3.py --validate [slots]
    python -I dataprep_staging/n3_piles/prereg_n3.py --write --mapping-dir D --g0-result F \
        --leveling-product P --expect-self-sha <sha256 of this file>     (the lead only)
        [--g0-fail-mode level-only --ruling <ruling json> --ruling-id <id>]   (user ruling)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0_n3 as G0  # noqa: E402
import n3lib as L  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
PENDING = "PENDING"
G0_REPS = 500                      # g0_n3.REPS: a full run, never a smoke or reduced run
FROZEN = tuple(f"{L.E}/{n}" for n in ("n3lib.py", "g0_n3.py", "build_mapping.py",
                                      "well_columns.py", "build_v3_manifest.py",
                                      "export_leveling_competitor.py", "score_n3.py",
                                      "prereg_n3.py")) + (L.SHARED_SCORER,)
RULINGS_DIR = "results/dataprep/rulings"
RULINGS_SCHEMA = "dataprep-rulings/1"
# the builder's own full G0 run (outside the repository; not canonical), disclosed because its
# failure was known before the canonical run (review 2026-11-01)
SCRATCH_G0 = {
    "where": "builder scratch directory outside the repository, 2026-11-01 (not canonical)",
    "result_sha256": "5deadeba164acf83760719301c2dd14fef326b8f0f10e16374c987fc57b759e2",
    "model_side_sha256": "ca092bae536f424ae35ef1fe743f923aee9c34f38450c763da8d81380996fdb4",
    "code": {"dataprep_staging/n3_piles/g0_n3.py":
             "e4f2e4fd7d1ac8f0765e0edb4aed33ec7827dbbab252ae97277042a6c65101d0",
             "dataprep_staging/n3_piles/n3lib.py":
             "3f5094ca42dac88a924fe5a1e1f4bcdb366a483645a067d7ba907172a2455ee2"},
    "reps": 500, "smoke": False,
    "outcome": "failed at 12 and 24 months (decision: conversion report-only); at rho = 1 "
               "the median rho_hat was 0.47 (sd 0.75 m, phi 0.5), 0.82 (sd 0.75, phi 0.9), "
               "0.18 (sd 1.5, phi 0.5), 0.54 (sd 1.5, phi 0.9) at 12 months; coverage "
               "0.002 / 0.40 / 0.0 / 0.008: errors-in-variables attenuation",
    "after": "the decision rule was then routed through N2's shared scorer (review fix; "
             "verdict labels of non-finite and negative-point cases only) and model_side.json "
             "gained the column-pairing fields, so the canonical G0 reruns on the new shas"}
G0_FROZEN = ("SD_GRID {0.75, 1.5} m, RHOS {0.4, 1, 2}, REPS 500, the pass thresholds (0.80 / "
             "0.10 / coverage 0.85) and the 3-month block bootstrap B 2000 are the judge's; "
             "PHI_GRID {0.5, 0.9} was declared in g0_n3.py before its first run; the estimator "
             "(pile-FE OLS slope) and N2's shared decision rule are the scorer's. All are "
             "frozen: changing any of them after the known failure (e.g. an errors-in-"
             "variables or IV estimator) needs a user ruling and a fresh, disclosed G0 run")
CODE_AT_REGISTRATION = ("hydrophysics/twin/report_extract.py", "hydrophysics/twin/mechanism.py",
                        "hydrophysics/twin/leveling.py", "hydrophysics/twin/column_step.py",
                        "dataprep_staging/harness/prereg3.py", "dataprep_staging/common/guard.py",
                        "dataprep_staging/common/manifest.py", "dataprep_staging/common/uses.py")
ENGINEERING_RESTORE = ("a git-tracked module listed in code_at_registration_report_only may be "
                       "restored to its registered sha256 before the scoring run as an "
                       "engineering step, logged in the report; frozen_code is never edited")
PILE_WORDS = ("pile", "piles", "deep_bench", "dbm_", "SJES", "TKJS", "NTUH", "溪州", "土庫",
              "虎尾")
SCOPE_WORDS = ("pile", "deep_bench", "dbm")
UNIT_MEMBERS = {
    "description": "WRA printed deep-pile months (table 3 of the 地層下陷監測整合分析月報; "
                   "007001 table 1 as the declared second source) 2026-12..2028-11 at the "
                   "two Yunlin piles anchored at >= 300 m: 土庫國中 400 m (300 m fallback "
                   "sensor) and 中科虎尾 300 m",
    "ids": ["dbm_printed:土庫國中:400", "dbm_printed:土庫國中:300", "dbm_printed:中科虎尾:300"],
    "window": "2026-12..2028-11"}
UNIT_XZ_MEMBERS = {
    "description": "WRA printed deep-pile months 2026-12..2027-11 at 溪州國小 300 m (Changhua; "
                   "report-only arm)",
    "ids": ["dbm_printed:溪州國小:300"], "window": "2026-12..2027-11"}


def _sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _rec(p) -> dict:
    p = Path(p)
    p = p if p.is_absolute() else REPO / p
    return {"path": rel_to_repo(p), "sha256": L.sha256(p)}


# ----------------------------------------------------------------------------- slots
def mapping_block(d: Path | None) -> dict:
    if d is None:
        return {"status": PENDING}
    d = Path(d)
    files = {n: _rec(d / n) for n in ("mapping.json", "model_side.json") if (d / n).is_file()}
    if len(files) < 2:
        return {"status": "incomplete", "files": files}
    ms = json.loads((d / "model_side.json").read_text(encoding="utf-8"))
    mp = json.loads((d / "mapping.json").read_text(encoding="utf-8"))
    st = "ok"
    if ms.get("mapping_sha256") != files["mapping.json"]["sha256"]:
        st = "model_side does not name this mapping.json"
    elif ms.get("forward", {}).get("sha256") != L.FORWARD_SHA:
        st = "model_side was not built on the registered forward"
    elif not ms.get("informative_model_side_pass"):
        st = "model-side informative gate fails (RMS dD < 0.3 m or increment < 0.1 cm)"
    elif "column_pairing" not in ms or "g_mod_hindcast_paired" not in ms:
        st = "model_side lacks the column-pairing fields (rebuild with this build_mapping.py)"
    piles = {k: {"site": v["site"], "fallback_site": v.get("fallback_site"),
                 "nest": v["nest"], "layer_fill": v["layer_fill"], "role": v["role"]}
             for k, v in mp["piles"].items()}
    xz = ms.get("xizhou_r1c") or {}
    return {"status": st, "files": files, "piles": piles,
            "g_mod_12": ms["g_mod_12"], "g_mod_24": ms["g_mod_24"],
            "g_mod_definition": ms.get("g_mod_definition"),
            "column_pairing": ms.get("column_pairing"),
            "g_mod_hindcast_paired": {k: (ms.get("g_mod_hindcast_paired") or {}).get(k)
                                      for k in ("g_mod", "n", "months", "role")},
            "informative_model_side": ms["informative_model_side"],
            "zr_level": ms["zr_level"], "depth_convention": ms["depth_convention"],
            "xizhou_r1c": {k: xz.get(k) for k in ("forward", "column", "k_mod_2027",
                                                  "k_mod_definition",
                                                  "discontinuity_threshold_cm", "rehearsal")}}


def g0_block(f: Path | None, mapping: dict) -> dict:
    if f is None:
        return {"status": PENDING, "decision": None}
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    out = {"result": _rec(f), "decision": r.get("decision"), "status": "ok",
           "pass_12": r.get("horizon_12", {}).get("pass"),
           "pass_24": r.get("horizon_24", {}).get("pass"),
           "fails_12": r.get("horizon_12", {}).get("fails", [])[:8],
           "fails_24": r.get("horizon_24", {}).get("fails", [])[:8],
           "parameters": r.get("parameters")}
    want = G0.code_shas()
    par = r.get("parameters") or {}
    if r.get("id") != "N3.G0.v1" or r.get("smoke") is not False or \
            par.get("reps") != G0_REPS or par.get("boot_B") != L.BOOT_B:
        out["status"] = "not a full N3.G0.v1 run"
    elif r.get("code") != want:
        out["status"] = "G0 was run with other code (g0_n3.py / n3lib.py / shared scorer)"
    elif mapping.get("files") and r.get("model_side", {}).get("sha256") != \
            mapping["files"]["model_side.json"]["sha256"]:
        out["status"] = "G0 was run on another model_side.json"
    return out


def g0_attempts(g0_result: Path | None, archive: Path | None = None) -> dict:
    """Every G0 attempt: the archived runs (run_n3.sh archive: <archive>/<UTC stamp>/g0/
    result.json + REASON.txt) and the builder's scratch run. Model-only files."""
    runs = []
    if g0_result is not None:
        archive = Path(archive) if archive else Path(g0_result).resolve().parent.parent / \
            "archive"
    if archive is not None and Path(archive).is_dir():
        for d in sorted(p for p in Path(archive).iterdir() if p.is_dir()):
            res = sorted(d.rglob("result.json"))
            why = (d / "REASON.txt").read_text(encoding="utf-8").strip() \
                if (d / "REASON.txt").is_file() else None
            rec = {"archive": d.name, "reason": why}
            for f in res:
                try:
                    r = json.loads(f.read_text(encoding="utf-8"))
                except ValueError:
                    r = {}
                rec.setdefault("results", []).append({
                    "sha256": L.sha256(f), "smoke": r.get("smoke"),
                    "pass_12": (r.get("horizon_12") or {}).get("pass"),
                    "pass_24": (r.get("horizon_24") or {}).get("pass"),
                    "primary_horizon": (r.get("decision") or {}).get("primary_horizon"),
                    "code": r.get("code")})
            runs.append(rec)
    return {"archived": runs, "n_archived": len(runs), "scratch": SCRATCH_G0,
            "frozen": G0_FROZEN}


def ruling_block(path: Path | None, rid: str | None, rulings_dir: Path | None = None) -> dict:
    """The user ruling that authorises the level-only fallback (status 'ok' or why not)."""
    if path is None or not rid:
        return {"status": "absent"}
    p = Path(path)
    p = p if p.is_absolute() else REPO / p
    root = Path(rulings_dir) if rulings_dir else REPO / RULINGS_DIR
    out = {"status": "ok", "id": rid}
    if not p.is_file():
        return {"status": f"no ruling file {p.name}"}
    if root.resolve() not in p.resolve().parents:
        return {"status": f"the ruling file is not under {RULINGS_DIR}"}
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {"status": "the ruling file is not JSON"}
    out.update({"file": rel_to_repo(p) if REPO in p.resolve().parents else p.name,
                "sha256": L.sha256(p), "date": r.get("date")})
    hit = [x for x in r.get("rulings") or [] if isinstance(x, dict) and x.get("id") == rid]
    if r.get("schema") != RULINGS_SCHEMA or r.get("ruling_by") != "user":
        out["status"] = f"not a user ruling record ({RULINGS_SCHEMA}, ruling_by 'user')"
    elif len(hit) != 1:
        out["status"] = f"ruling id {rid!r} not found once in the record"
    elif not any(L.ID in str(a) for a in hit[0].get("applies_to") or []):
        out["status"] = f"ruling {rid!r} does not name {L.ID} in applies_to"
    elif not str(hit[0].get("decision") or "").strip():
        out["status"] = f"ruling {rid!r} has no decision text"
    else:
        out.update({"question": hit[0].get("question"), "decision": hit[0]["decision"],
                    "applies_to": hit[0]["applies_to"]})
    return out


def leveling_block(d: Path | None) -> dict:
    if d is None:
        return {"status": PENDING}
    d = Path(d)
    from common.manifest import load_manifest

    m, _ = load_manifest(d)
    files = {n: _rec(d / n) for n in ("MANIFEST.json", "competitor.csv", "validation.json")}
    st = "ok"
    if "baseline" not in m.get("allowed_uses", []) or not m.get("gate", {}).get("passed"):
        st = "the product does not grant baseline or its gate failed"
    elif m.get("fit_window_end", "9999") > "2021-12":
        st = "fit_window_end after 2021-12"
    return {"status": st, "dir": rel_to_repo(d), "files": files,
            "product": f"{m.get('product')}/{m.get('version')}"}


def pile_looks(prereg_dir=None) -> dict:
    """Every logged look whose units or data scope name a pile (looks.jsonl metadata only);
    'late': a pile-data scope (pile, deep_bench, dbm) reaching 2026-02 or later (refuses
    --write); 'unbounded': a pile-data scope prereg3 cannot bound (disclosed; prereg3's
    membership rule still applies)."""
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    looks = prereg3.read_chained(d / "looks.jsonl") if (d / "looks.jsonl").exists() else []
    hits, late, unb = [], [], []
    lim = 12 * 2026 + 1                              # 2026-02 as a month index
    for lk in looks:
        s = json.dumps({k: lk.get(k) for k in ("units", "data_scope")}, ensure_ascii=False)
        if not any(w in s for w in PILE_WORDS):
            continue
        b = {"time": lk.get("time"), "wp": lk.get("wp"), "purpose": lk.get("purpose"),
             "data_scope": lk.get("data_scope"), "units": (lk.get("units") or [])[:4]}
        hits.append(b)
        if not any(k in str(lk.get("data_scope")) for k in SCOPE_WORDS):
            continue                     # a pile site named in a unit of another data type
        w = prereg3.scope_windows(lk.get("data_scope"))
        if w is None:
            unb.append(b)
        elif max(x[1] for x in w) >= lim:
            late.append(b)
    return {"n": len(hits), "looks": hits, "late": late, "unbounded": unb}


# ----------------------------------------------------------------------------- the document
NOT_REGISTERED = "not_registered"


def roles(g0: dict, fail_mode: str) -> dict:
    """Roles fixed by G0. G0 failed (or not run) and no level-only ruling: every test
    'not_registered' (the judge's rule; ``--write`` refuses)."""
    dec = g0.get("decision") or {}
    h = dec.get("primary_horizon")
    if h is None and fail_mode != "level-only":
        tests = (L.TEST_CONV, L.TEST_LEVEL, L.TEST_CONV24, L.TEST_LEVEL24, L.TEST_XZ)
        return {**dict.fromkeys(tests, NOT_REGISTERED), "g0_fail_mode": fail_mode}
    return {L.TEST_CONV: "gating" if h == 12 else "report_only",
            L.TEST_CONV24: "gating" if h == 24 else "report_only",
            L.TEST_LEVEL: "gating", L.TEST_LEVEL24: "report_only",
            L.TEST_XZ: "report_only", "g0_fail_mode": fail_mode if h is None else None}


def _test(tid, scope, units: dict, role, **kw) -> dict:
    return {"test_id": tid, "data_scope": scope, "fresh_units": list(units),
            "external_units": dict(units), "seed": L.BOOT_SEED, "role": role,
            "label": "prospective: the window's pile months did not exist at registration",
            **kw}


def heads_units(mp: dict) -> dict:
    """N3's own head units: the nest wells' months outside N2's / N2C's fresh units
    (2026-11..2027-10), which N3 reads and which no other registration seals."""
    piles = mp.get("piles") or {}

    def ids(keys):
        return sorted({w["sid"] for k in keys for w in (piles.get(k) or {}).get("nest", [])})

    def members(keys, window, what):
        i = ids(keys)
        base = {"description": what, "window": window}
        return {**base, "ids": i} if i else {**base, "none": "mapping pending"}
    return {
        L.UNIT_HEADS_YL: members(L.PRIMARY, L.HEADS_WINDOW_YL,
                                 "window heads of the Yunlin pile nests (wells within 1.5 km "
                                 "of 土庫國中 and 中科虎尾, mapping.json) in the months N2's "
                                 "fresh unit does not cover: 2026-09..2026-10 (dD and lags "
                                 "of the first window months) and 2027-11..2028-11 (the last "
                                 "12-month month and the 24-month continuation)"),
        L.UNIT_HEADS_XZ: members(("xizhou",), L.HEADS_WINDOW_XZ,
                                 "window heads of the 溪州國小 nest (wells within 3 km) in "
                                 "2027-11, the season's last month, outside N2C's fresh unit")}


CONV_METRIC = {
    "observed": "per primary pile p and window month m: c(p, m) = the two-read-verified printed "
                "monthly compression (cm, table 3; flag 0 of deep_benchmarks_printed v3) at the "
                "declared sensor (土庫國中 400 m; the 300 m sensor only if 400 m has fewer than "
                f"{L.MIN_MONTHS} months and 300 m has at least that many; never mixed); "
                "dD(p, m) = the nest's month-mean drawdown change (n3lib.nest_dd: per layer the "
                "mean head change of the nest wells screened there, unscreened layers filled "
                "from the nearest screened layer by median screen depth, equal layer weights "
                "as the column's driver) from the window heads product granting "
                "prospective_scorer (N2's feed; flags 0-1); fallback: the printed table-3 well "
                "columns (well_columns.py, two-read accepted) only for a pile whose primary dD "
                f"has fewer than {L.MIN_MONTHS} months",
    "g_obs": "pooled OLS slope of c on dD with a pile fixed effect (n3lib.fe_slope)",
    "g_mod": "the same regression on the frozen ZR model (datum_gate_innov, coupled_leveling "
             "column, ensemble mean) at the pile cells, against the DISPLAYED (restarted) ZR "
             "heads at the nest wells' cells and layers: the column increment per metre of "
             "DISPLAYED ZR head change. The column itself is driven by the free projection "
             "(column_heads 'free', fix A2), which the archive does not save, so these heads "
             "are not the column's driver (model_side.column_pairing); a frozen NUMBER",
    "rho": "g_obs / g_mod",
    "interval": f"90 % percentile interval of g_obs from a {L.BOOT_BLOCK}-month moving-block "
                f"bootstrap within pile, B {L.BOOT_B}, seed {L.BOOT_SEED}; rho interval = "
                "that interval / g_mod",
    "report_only": ["distributed lag 0-2 (sum of the three FE coefficients) / g_mod",
                    "slope split by the sign of dD (falling vs rising head)",
                    "seasonal compaction-calibration slope: per-pile-demeaned observed "
                    "increments on the ZR model increments (head path x conversion; old T2)",
                    "g_mod on the cell's own layer-mean head (report-only)",
                    "rho_paired = g_obs / g_mod_hindcast_paired: the column's slope on the "
                    "hindcast heads that drove it (2016-06..2022-12, gain 0; model_side; "
                    "report-only comparator for the column-pairing gap)"]}
CONV_RULE = {
    "shared_code": f"{L.SHARED_SCORER} decide() (N2's rule, ratio branch) on rho and its "
                   "interval, via n3lib.conversion_verdict; " + L.RMSE_CLAUSE,
    "conversion_supported": "the 90 % rho interval excludes 0, rho > 0 AND rho lies in "
                            "[1/1.5, 1.5] (N2's 'response_supported', ratio_band 1.5)",
    "direction_supported_size_not": "the interval excludes 0 and rho > 0, rho outside "
                                    "[1/1.5, 1.5]",
    "not_supported": "the interval includes 0, rho <= 0, or an estimate is not finite",
    "not_informative": "the informative gate fails, or g_mod is not usable",
    "differences_from_round5_draft": "the draft's own rule called a non-finite estimate "
                                     "'not_informative' and a negative point with a positive "
                                     "interval 'direction_supported_size_not'; the shared "
                                     "rule calls both 'not_supported' (as N2)"}
CONV_GATE = (f"model side (computed now, in 'model_side'): at each primary pile RMS of the "
             f"model dD >= {L.MIN_RMS_DD_M} m and RMS of the model increment >= "
             f"{L.MIN_RMS_INC_CM} cm; observed: >= {L.MIN_MONTHS} (24 months: "
             f"{L.MIN_MONTHS24}) window months with a verified pile value AND a nest dD at "
             "EACH primary pile by the no_later_than date")
LEVEL_METRIC = {
    "observed": "per primary pile, the window compression (cm): the printed trailing-12 value "
                "Y(2027-11) (two-read verified, same sensor); if absent, the sum of the 12 "
                "verified monthly values; otherwise none. 24 months: Y(2027-11) + Y(2028-11), "
                "else the 24 summed months",
    "M": "the frozen ZR column at the pile cell, cumulative from 2026-11 (frozen numbers, "
         "'model_side.zr_level'); unscaled (depth convention)",
    "competitors": "zero, and the pre-origin leveling-cell rate x the window length "
                   "(n3_leveling_cell_rate product: benchmarks in the pile cell, else within "
                   "1.5 km, median 2016-01..2021-12 rate; baseline use, window_end 2021-12). "
                   "Printed-pile persistence is NOT used (ruling 2026-10-18 (b); user ruling "
                   "2026-11-01)",
    "tolerance": "none: no absolute tolerance clause (none derivable from pre-origin "
                 "quantities alone)",
    "report_only": ["the 0.8 x M variant (anchor-depth share)",
                    "the other observed construction (sum vs trailing-12) as a check"]}
LEVEL_RULE = {
    "level_supported": "|M - obs| < |0 - obs| AND |M - obs| < |L - obs| at BOTH primary piles",
    "level_not_supported": "otherwise",
    "not_evaluable": "a primary pile has no observed window value or no leveling competitor "
                     "by the no_later_than date"}
XZ_METRIC = {
    "estimator": "season-aggregate (n3lib.season_coefficients): rebound over Jun-Nov 2027 "
                 "(minus the summed verified printed monthly compression, cm) / the season "
                 "head rise (minus the summed nest dD, m), nest = N2C's wells within 3 km; "
                 "t-interval df = n - 1 (n = 1 season: point only)",
    "model": "the same estimator on R1c's frozen ZR (datum_gate_ccz.npz) at the 溪州 cell "
             "and nest: R1c's column rebound per metre of DISPLAYED (restarted) ZR head rise "
             "(the column runs on the unsaved free projection, column_heads 'free'); its "
             "value and the informativeness rehearsal are in 'model_side' (frozen ZR heads "
             "only, never X1 or PR); R1c's paired hindcast seasons 2019-22 "
             "(k_mod_hindcast_paired) are a report-only comparator",
    "reset_rule": "a season is dropped (never bridged) if a Jun-Nov month is missing, not "
                  "verified (flags 8/9), printed at another depth, or |monthly| exceeds the "
                  "frozen threshold max(1.0 cm, 5 x the model's largest monthly increment)",
    "reads": "only after N2C's step-(2) report exists (session order); the heads are N2C's "
             "fresh unit's wells, read in that session"}


def doc(mapping_dir: Path | None = None, g0_result: Path | None = None,
        leveling_product: Path | None = None, fail_mode: str = "refuse",
        prereg_dir=None, ruling: Path | None = None, ruling_id: str | None = None,
        g0_archive: Path | None = None, rulings_dir: Path | None = None) -> dict:
    mp = mapping_block(mapping_dir)
    g0 = g0_block(g0_result, mp)
    g0["attempts"] = g0_attempts(g0_result, g0_archive)
    lv = leveling_block(leveling_product)
    rl = roles(g0, fail_mode)
    rb = ruling_block(ruling, ruling_id, rulings_dir)
    hu = heads_units(mp)
    looks = pile_looks(prereg_dir)
    frozen = {p: L.sha256(REPO / p) for p in FROZEN if (REPO / p).is_file()}
    code_reg = {p: L.sha256(REPO / p) for p in CODE_AT_REGISTRATION if (REPO / p).is_file()}
    binds = {"forward": {"path": L.FORWARD, "sha256": L.FORWARD_SHA},
             "column": {"path": L.COLUMN, "sha256": L.COLUMN_SHA}}
    inputs = sorted(set(frozen) | {L.FORWARD, L.COLUMN}
                    | {r["path"] for r in (mp.get("files") or {}).values()}
                    | ({g0["result"]["path"]} if "result" in g0 else set())
                    | {r["path"] for r in (lv.get("files") or {}).values()}
                    | ({rb["file"]} if rb.get("status") == "ok" and "/" in rb["file"]
                       else set()))
    common = {"metric_shared_code": f"{L.E}/n3lib.py",
              "decision_shared_code": L.SHARED_SCORER}
    u_pile = {L.UNIT: UNIT_MEMBERS}
    u_conv = {**u_pile, L.UNIT_HEADS_YL: hu[L.UNIT_HEADS_YL]}
    u_xz = {L.UNIT_XZ: UNIT_XZ_MEMBERS, L.UNIT_HEADS_XZ: hu[L.UNIT_HEADS_XZ]}
    t_conv = _test(L.TEST_CONV, L.SCOPE_CONV12, u_conv, rl[L.TEST_CONV],
                   claim="C31: compaction per metre of head change at the 2 Yunlin deep-pile "
                         "cells", window=list(L.WINDOW), metric=CONV_METRIC,
                   decision_rule=CONV_RULE, informative_gate=CONV_GATE, **common)
    t_conv24 = _test(L.TEST_CONV24, L.SCOPE_CONV24, u_conv, rl[L.TEST_CONV24],
                     claim="C31 (24-month continuation)", window=list(L.WINDOW24),
                     metric=CONV_METRIC, decision_rule=CONV_RULE,
                     informative_gate=CONV_GATE, **common)
    t_level = _test(L.TEST_LEVEL, L.SCOPE12, u_pile, rl[L.TEST_LEVEL],
                    claim="C09/C30 narrowed: pile-anchored status-quo compaction at the 2 "
                          "Yunlin pile cells", window=list(L.WINDOW), metric=LEVEL_METRIC,
                    decision_rule=LEVEL_RULE,
                    informative_gate="both primary piles have an observed window value and a "
                                     "leveling competitor", **common)
    t_level24 = _test(L.TEST_LEVEL24, L.SCOPE24, u_pile, rl[L.TEST_LEVEL24],
                      claim="C09/C30 narrowed (24-month continuation, report-only)",
                      window=list(L.WINDOW24), metric=LEVEL_METRIC, decision_rule=LEVEL_RULE,
                      informative_gate="as T-level, 24 months", **common)
    t_xz = _test(L.TEST_XZ, L.SCOPE12, u_xz, rl[L.TEST_XZ],
                 claim="C13c/C27 wording only (Changhua elastic coefficient at 溪州); never "
                       "gates; no Changhua subsidence lever under any outcome",
                 window=list(L.WINDOW), metric=XZ_METRIC,
                 decision_rule={"report_only": "k_obs, k_mod and k_obs / k_mod are reported; "
                                               "no verdict"},
                 informative_gate="at least one Jun-Nov 2027 season kept by the reset rule",
                 **common)
    session = [
        "(1)-(10): the order declared in E02c.prospective_2026-11.v1 (N2C step (1) first; N2 "
        "Yunlin primary; N2C step (2); ...; N3F)",
        f"(11) {L.ID} T-conversion and T-level: only after N2's score look "
        f"({L.N2_ID} / {L.N2_TEST}) AND N2C's step-(1) score look are in looks.jsonl; N2's "
        "head unit is an external input of N3, never a fresh unit",
        f"(12) {L.ID} 溪州 arm: only after N2C's step-(2) report exists (appended after N2C "
        "by user ruling 2026-11-01)",
        "(13) 24-month continuation: a separate session, not before 2029-01-15, same order "
        "rules"]
    conv_gating = rl[L.TEST_CONV] == "gating" or rl[L.TEST_CONV24] == "gating"
    unregistered = rl[L.TEST_LEVEL] == NOT_REGISTERED
    page = {
        "before_scoring": {"status": "N3 not registered (G0 failed or not run, no level-only "
                                     "user ruling): see default_without_ruling"}
        if unregistered else {
            "C31": "new claim, status D: 'compaction per metre of head change at the 2 Yunlin "
                   "deep-pile cells: decided by N3 (not before 2028-01-15)'"
                   + ("" if conv_gating else
                      "; G0 failed and N3 is registered level-only on a user ruling "
                      f"({ruling_id}): C31 is N, 'conversion not identifiable at the 2 Yunlin "
                      "deep-pile cells by design (N3 G0); reported only, not tested'"),
            "C09_C30": "D sub-label 'pile-anchored status-quo compaction at 2 Yunlin cells "
                       "decided by N3 (not before 2028-01-15)'",
            "C27": "names N3 as the conversion decider" if conv_gating else
                   "conversion factor 'not identifiable at the 2 Yunlin deep-pile cells by "
                   f"design (N3's synthetic check G0); reported only, not tested' (user "
                   f"ruling {ruling_id})",
            "C15": "untouched (context link to C31 only)"},
        "default_without_ruling": "if G0 fails and no user ruling authorises level-only, N3 "
                                  "is not registered: C31 reads 'untested: G0 failed', C27 "
                                  "'conversion factor: untested (no registered test)', and "
                                  "C09/C30 carry no N3 sub-label",
        "conversion": {
            "joint": "N2 'response_supported' AND N3 'conversion_supported' (gating role): "
                     "C31 -> S ('seasonal elastic conversion at 2 pile cells, prospective'); "
                     "ledger line 'both model links tested at the seasonal scale at 2 pile "
                     "cells'; C03 stays W ('links tested, cut size untested'); no cut number "
                     "becomes S",
            "conversion_supported_only": "N2 not 'response_supported': C31 N, 'conversion "
                                         "consistent at 2 pile cells (prospective); head "
                                         "link: N2 <verdict>'",
            "direction_supported_size_not": "C31 N: 'sign supported; size outside 1/1.5-1.5 "
                                            "(rho = X)'; no correction factor (ruling 4)",
            "not_supported": "C31 N/W decided; no correction factor (ruling 4)",
            "not_informative": "C31 N, 'undecided (not informative)'",
            "report_only_role": "any verdict of a report-only conversion line is evidence "
                                "text only; C31 stays N"},
        "level": {
            "level_supported": "C09/C30 narrowed sub-claim 'status-quo compaction at the 2 "
                               "Yunlin pile cells' passes (prospective). User ruling "
                               "2026-11-01: only C31 can become S, so the sub-claim reads "
                               "'passed N3 (prospective)' at status N; fan-wide C09/C30 "
                               "unchanged",
            "level_not_supported": "the sub-claim is decided N with the verdict printed",
            "not_evaluable": "'undecided (WRA reports not published)'"},
        "xizhou": "C13c and C27 wording only; never gates; no Changhua subsidence lever under "
                  "any outcome (N2C 'never')",
        "never": "no cut number, no coastal, long-run or Changhua subsidence claim follows "
                 "from any N3 verdict"}
    disclosures = {
        "priors_both": "rings: seasonal compaction per metre 3.4 mm/m against the column's "
                       "9.3 mm/m (twin.mechanism, rings to 2025, after the fact; points to a "
                       "fail); 2024-25 piles >= 200 m: column R2 0.92 (results/twin/"
                       "deep_pile_holdout; used to choose the column, E06, U7; points to a "
                       "pass). Neither set a threshold here",
        "earlier_pile_looks": looks["looks"][:120],
        "n_earlier_pile_looks": looks["n"],
        "pile_looks_unbounded_scope": looks["unbounded"],
        "report_extraction_scope": "extractions of the June/July 2027 and 2028 WRA monthly "
                                   "reports (007001/007003) transcribe the pile tables only, "
                                   "never the statement (綜合說明) pages that give that year's "
                                   "leveling result (kept sealed for N3SL, scored after N3)",
        "site_selection": "three of the on-grid piles stand at schools whose rings were a "
                          "Stage-2 target through 2022; the 2024-25 pile scores were used to "
                          "choose between columns (U7, 2026-10-21)",
        "g0": "G0 (synthetic, model-only) fixed the conversion test's role before "
              "registration; result hashed under 'g0'; every attempt (archived runs and the "
              "builder's scratch run, whose failure was known before the canonical run) is "
              "listed under g0.attempts; " + G0_FROZEN,
        "column_pairing": "both forward archives were run with column_heads 'free' (fix A2): "
                          "after the 2022-12 origin heads_mean holds the DISPLAYED, restarted "
                          "heads, while the column runs on the free projection from the "
                          "hindcast's end state, which is not saved. g_mod (and R1c's k_mod) "
                          "is therefore the column increment per metre of DISPLAYED ZR head "
                          "change, not per metre of its driver; the window heads are never "
                          "described as the column's driver. The paired hindcast slope "
                          "(2016-06..2022-12, gain 0) is a declared report-only comparator "
                          "(model_side.g_mod_hindcast_paired; scorer rho_paired_hindcast)",
        "multiplicity": "N3 holds 5 tests on the pile unit (2 gating at most); N2's report-only "
                        "pile arm reads the same months as report text; no family-wise claim",
        "one_role": "user ruling 2026-11-01: the pile months move from N2's report-only arm to "
                    "N3's gating role; N2's arm stays report text",
        "availability": "existence of reports after 115年1月 checked only by HEAD or listing "
                        "requests, logged as availability_check; no pile month after 2026-01 "
                        "is fetched or read before scoring",
        "transcription": "twin.report_extract two-model read (PDF and layout text, a third "
                         "read settles a disagreement; verified = two agree); products built "
                         "by build_v3_manifest.py (future months only); the 007001 彰雲 "
                         "monthly report (table 1) is the declared second source for a month "
                         "007003 does not print, never mixed within a month",
        "sign": "value = compression (deep_benchmarks_printed MANIFEST; report_extract: "
                "negative = rebound); dD = minus the head change",
        "depth": mp.get("depth_convention", PENDING),
        "ruling_2026_10_18": "pre-origin printed piles (validator, after_the_fact_analysis "
                             "only) enter no gate, threshold or competitor here",
        "xizhou_r1c": "R1c's six-zone column was suspended as a subsidence fallback "
                      "(2026-10-21) and carries the post-2022 uplift artefact; the 溪州 arm "
                      "is report-only; 溪州 (SJES) was looked at many times before "
                      "(S1a/S1b/S2/S3/X5)",
        "lag": "lag 0 only in the decision; delayed drainage attenuates g_obs by timing, and "
               "drawdown below past lows is inelastic in reality but elastic in the page "
               "model; lags 0-2 and the sign split are report-only"}
    d = {
        "id": L.ID, "wp": L.WP, "kind": "test",
        "title": "N3: prospective pile tests at the 2 Yunlin deep piles (conversion per metre "
                 "of head change, level against zero and the leveling-cell rate) with a "
                 "report-only 溪州 arm, 2026-12..2027-11 (+24 months)",
        "why": "No test decides the column's compaction per metre of head change or its "
               "status-quo level after 2022; the printed WRA deep piles are the only "
               "continuing instrument (user ruling 2026-11-01, fourth set).",
        "tests": [t_conv, t_level, t_conv24, t_level24, t_xz],
        "roles": rl,
        "binds": {**binds, "code_at_registration_report_only": code_reg,
                  "engineering": ENGINEERING_RESTORE},
        "model_side": mp, "g0": g0, "leveling_competitor": lv,
        "frozen_code": frozen,
        "scoring": {
            "not_before": "2028-01-15 (12 months); 2029-01-15 (24 months)",
            "no_later_than": "2029-06-30 (12 months); 2030-06-30 (24 months); later: "
                             "not_informative / not_evaluable",
            "precondition": "deep_benchmarks_printed v3 (future months only) and the window "
                            "heads product both grant prospective_scorer, checked from their "
                            "MANIFESTs alone before any read; otherwise not evaluable, nothing "
                            "read, no look",
            "run_order": "(1) MANIFEST checks; (2) order checks; (3) prereg3.assert_can_score "
                         "per test (logs the looks); (4) only then pile and head rows are read",
            "same_session_order": session,
            "scorer": f"{L.E}/score_n3.py score --horizon 12 --part piles --piles-product "
                      "<v3> --heads-product <window heads> --out <dir> (step 11); then --part "
                      "xizhou --n2c-report <N2C report> --out <dir2> (step 12); 24 months: "
                      "--horizon 24 --part piles, not before 2029-01-15; each test's score "
                      "look is logged once by prereg3.assert_can_score"},
        "sealed_until_scoring": "every pile month after 2026-01 and every window head; no "
                                "statistic of the window is computed before the scoring look",
        "page_mapping": page, "disclosures": disclosures,
        "what_a_pass_supports": "seasonal elastic conversion at 2 pile cells (conversion); "
                                "pile-anchored status-quo compaction at 2 pile cells (level); "
                                "not the 1-3 yr or the inelastic response, not the coast, the "
                                "long run, Changhua subsidence or any cut number",
        "attempt_policy": "User rule 2026-10-23: engineering failures (crash, bug, missing "
                          "file) may be fixed and re-run before the scoring look, each logged; "
                          "each test is scored ONCE; a failed, inconclusive, not-informative "
                          "or not-evaluable result is reported as such and never re-scored on "
                          "these units; the report-only lines and the 24-month continuation "
                          "are fixed now and never replace the primary; after a failure at "
                          "most ONE follow-up registration (new test ids, 'post-failure, "
                          "exploratory', never validation) that cannot reuse this window; the "
                          "final report lists every attempt",
        "rulings": {
            "2026-11-01 fourth set (rank 6)": "APPROVED with the 溪州 report-only arm: a "
                "future-month deep_benchmarks_printed v3 may list prospective_scorer; the pile "
                "months move from N2's report-only arm to N3's gating role (N2's arm stays as "
                "report text); the pre-origin leveling-cell rate replaces printed-pile "
                "persistence; only C31 can become S, no cut number; 溪州 arm after N2C; window "
                "2026-12, registered by 2026-11-30T16:00Z; G0 first",
            "2026-10-18 (b)": "pre-origin printed piles: validator and after_the_fact_analysis "
                              "only",
            "2026-10-29 (4)": "no calibrated correction factor follows from any verdict",
            **({f"G0 level-only ({ruling_id})": rb} if rb.get("status") == "ok" else {})},
        "zones": NA("no fit; the frozen forward run's own zones"),
        "priors": disclosures["priors_both"],
        "burn_in": NA("no fit"), "folds": NA("prospective window"),
        "fresh_units": [L.UNIT, L.UNIT_XZ, L.UNIT_HEADS_YL, L.UNIT_HEADS_XZ],
        "thresholds": {"ratio_band": L.RATIO_BAND, "level": L.LEVEL, "boot_B": L.BOOT_B,
                       "boot_block": L.BOOT_BLOCK, "boot_seed": L.BOOT_SEED,
                       "min_rms_dd_m": L.MIN_RMS_DD_M, "min_rms_inc_cm": L.MIN_RMS_INC_CM,
                       "min_months": L.MIN_MONTHS, "min_months_24": L.MIN_MONTHS24,
                       "nest_radius_m": L.NEST_RADIUS_M, "season": list(L.SEASON),
                       "min_season_rise_m": L.MIN_SEASON_RISE_M},
        "gate_scores": NA("no weighted gate; each decision rule is reported as it falls"),
        "primary_model": {"name": "the page model: stage3_datum_gate + coupled_leveling "
                                  "column, forward datum_gate_innov (ZR, ensemble mean)",
                          "forward_sha256": L.FORWARD_SHA, "column_sha256": L.COLUMN_SHA,
                          "xizhou_report_only_model": "R1c (datum_gate_ccz.npz)"},
        "external_inputs": {"heads": f"N2's window feed ({L.N2_ID} unit wells, 2026-11.."
                                     "2027-10; an input, not a fresh unit of N3; N2C's for "
                                     "溪州) via a product granting prospective_scorer; the "
                                     "nest months outside those units are N3's own fresh "
                                     f"units {L.UNIT_HEADS_YL} and {L.UNIT_HEADS_XZ}"},
        "inputs": inputs,
        "interpretation": "Whether the page column's compaction per metre of head change and "
                          "its status-quo compaction hold at the two Yunlin deep-pile cells "
                          "on fresh months. Nothing about cut sizes, the coast, the long run "
                          "or Changhua subsidence.",
        "written_by": f"{L.E}/prereg_n3.py",
    }
    return d


# ----------------------------------------------------------------------------- CLI
def problems_for_write(d: dict, expect_self_sha, fail_mode: str,
                       prereg_dir=None, ruling: dict | None = None) -> list[str]:
    ruling = ruling or {"status": "absent"}
    bad = []
    for k in ("model_side", "g0", "leveling_competitor"):
        if d[k].get("status") != "ok":
            bad.append(f"{k} slot: {d[k].get('status')}")
    dec = d["g0"].get("decision") or {}
    if d["g0"].get("status") == "ok" and dec.get("primary_horizon") is None:
        if fail_mode != "level-only":
            bad.append("G0 failed at 12 and 24 months: N3 is not registered as gating (judge); "
                       "level-only needs a user ruling (--g0-fail-mode level-only --ruling "
                       "<record> --ruling-id <id>)")
        elif ruling.get("status") != "ok":
            bad.append(f"--g0-fail-mode level-only needs a user ruling: {ruling.get('status')}")
    if any(v == NOT_REGISTERED for v in d["roles"].values()):
        bad.append("roles not registered (G0 decision missing or failed without a ruling)")
    if set(d["frozen_code"]) != set(FROZEN):
        bad.append(f"frozen code incomplete: {sorted(set(FROZEN) - set(d['frozen_code']))}")
    if L.sha256(REPO / L.FORWARD) != L.FORWARD_SHA:
        bad.append("forward npz sha changed")
    if L.sha256(REPO / L.COLUMN) != L.COLUMN_SHA:
        bad.append("column json sha changed")
    late = pile_looks(prereg_dir)["late"]
    if late:
        bad.append(f"pile looks whose scope reaches 2026-02 or later: {late[:3]}")
    self_sha = L.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != this script {self_sha}")
    if L._now() >= L.DEADLINE:
        bad.append(f"DEADLINE PASSED ({L.DEADLINE:%Y-%m-%dT%H:%MZ})")
    return bad


def check_blocking(d: dict, prereg_dir=None) -> list:
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    idx = prereg3.membership_index(pdir)
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            if not idx.known(u):
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
    return prereg3.blocking_looks(d["tests"], None, pdir, idx, test_time=L._now())


def main(argv=None, prereg_dir=None, rows_dir=None, rulings_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="write the registration (the lead)")
    ap.add_argument("--mapping-dir", type=Path, default=None)
    ap.add_argument("--g0-result", type=Path, default=None)
    ap.add_argument("--leveling-product", type=Path, default=None)
    ap.add_argument("--g0-fail-mode", choices=("refuse", "level-only"), default="refuse")
    ap.add_argument("--ruling", type=Path, default=None,
                    help=f"user ruling record under {RULINGS_DIR} (needed for level-only)")
    ap.add_argument("--ruling-id", default=None, help="the ruling's id inside the record")
    ap.add_argument("--g0-archive", type=Path, default=None,
                    help="archived G0 runs (default: <g0 result>/../../archive)")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc(a.mapping_dir, a.g0_result, a.leveling_product, a.g0_fail_mode,
            prereg_dir=prereg_dir, ruling=a.ruling, ruling_id=a.ruling_id,
            g0_archive=a.g0_archive, rulings_dir=rulings_dir)
    rb = ruling_block(a.ruling, a.ruling_id, rulings_dir)
    pl = pile_looks(prereg_dir)
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    block = check_blocking(d, prereg_dir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", _sha_text(d))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("pile looks:", pl["n"], "| reaching 2026-02+:", len(pl["late"]),
          "| unbounded scopes (disclosed):", len(pl["unbounded"]))
    print("slots: model_side", d["model_side"]["status"], "| g0", d["g0"]["status"],
          (d["g0"].get("decision") or {}).get("primary_horizon"), "| leveling",
          d["leveling_competitor"]["status"])
    print("roles:", json.dumps(d["roles"], ensure_ascii=False))
    print("g0 attempts: archived", d["g0"]["attempts"]["n_archived"], "| scratch run",
          d["g0"]["attempts"]["scratch"]["result_sha256"][:8], "| level-only ruling:",
          rb.get("status"))
    print("this script sha256:", L.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = problems_for_write(d, a.expect_self_sha, a.g0_fail_mode, prereg_dir, rb)
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
