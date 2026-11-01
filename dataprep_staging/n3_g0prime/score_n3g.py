"""Score N3Gprime.piles_conversion.v1 once (not before 2028-01-15; 24 months 2029-01-15).

Same session as N3, immediately after N3's step-(11) score looks (session step 11b). Order
of a real run: (1) the pile product and the window heads product grant prospective_scorer
(MANIFEST only; otherwise not evaluable, nothing read, no look); (2) N3's step-(11) score
looks for this horizon are logged, otherwise REFUSED; (3) prereg3.assert_can_score for this
horizon's test (logs the look); (4) only then pile and head rows are read. The rows are
N3's (score_n3 helpers: the declared sensor, verified months, nest dD by n3lib.nest_dd, the
printed well-column fallback); every number comes from g0p_lib.estimate_all with the
registered estimator and g_mod; the role is read from the registration.

Usage (repository root, the lead):
    python -I dataprep_staging/n3_g0prime/score_n3g.py score --horizon 12
        --piles-product results/dataprep/deep_benchmarks_printed/v3
        --heads-product <window heads product> [--stations <station metadata>]
        --out results/dataprep/reports/N3G/score_12m
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
for _p in (str(HERE), str(REPO / "dataprep_staging" / "n3_piles"),
           str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0p_lib as P  # noqa: E402
import score_n3 as S  # noqa: E402  (N3's row helpers, cited, never edited)

L = P.L
SCORER_VERSION = "1.0.0"
TEST_OF = {12: P.TEST_12, 24: P.TEST_24}
N3_LOOKS = {12: (L.TEST_CONV, L.TEST_LEVEL), 24: (L.TEST_CONV24, L.TEST_LEVEL24)}
WORDING = {k: v.replace("N3, prospective", "N3G, prospective")
           for k, v in S.WORDING.items()}


class ScoringRefused(S.ScoringRefused):
    """The scorer refuses to run (date, order or provenance)."""


def n3_looks_done(prereg_dir, h: int) -> dict:
    have = S._score_looks(prereg_dir)
    st = {t: (L.ID, t) in have for t in N3_LOOKS[h]}
    return {"n3_step11_looks": st, "ok": all(st.values())}


def rows(ms: dict, mapping: dict, pf: pd.DataFrame, heads: dict, span,
         fallback_changes: dict | None = None) -> tuple[dict, dict]:
    """N3's conversion rows (score_n3.conversion's construction) with their month labels and
    the model's nest dD at the same months (the instrument of iv_model_heads)."""
    need = L.MIN_MONTHS if span == L.WINDOW else L.MIN_MONTHS24
    cols = {"y": [], "x": [], "z": [], "m": [], "g": []}
    per = {}
    for key in L.PRIMARY:
        nest = mapping["piles"][key]["nest"]
        lay = {w["sid"]: w["layer"] for w in nest}
        ms_list = L.months(L.prev_month(span[0], 2), span[1])
        hh = {s: heads[s] for s in lay if s in heads}
        dd = L.nest_dd(hh, {s: lay[s] for s in hh}, ms_list) if hh else pd.Series(dtype=float)
        depth, inc = S.choose_sensor(pf, key, span)
        src = "window heads product"
        n_ok = int(sum(np.isfinite(dd.get(m, np.nan)) for m in inc.index))
        if n_ok < need and fallback_changes and key in fallback_changes:
            ch, lay2 = fallback_changes[key]
            dd = L.nest_dd_from_changes(ch, lay2, ms_list)
            src = "printed table-3 well columns (fallback)"
        w = ms["world"]["piles"][key]
        zmod = dict(zip(w["months"], w["dd_nest_m"], strict=True))
        kept = [(m, inc[m], dd.get(m, np.nan), zmod.get(m, np.nan)) for m in inc.index]
        kept = [r for r in kept if np.isfinite(r[2]) and np.isfinite(r[3])]
        per[key] = {"depth_m": depth, "n_months": len(kept), "head_source": src,
                    "pile_months_verified": int(len(inc))}
        for m, c, d0, z0 in kept:
            for k, v in zip(("m", "y", "x", "z", "g"), (m, c, d0, z0, key), strict=True):
                cols[k].append(v)
    return cols, per


def conversion(doc: dict, ms: dict, mapping: dict, pf: pd.DataFrame, heads: dict, h: int,
               fallback_changes: dict | None = None) -> dict:
    t = [x for x in doc["tests"] if x["test_id"] == TEST_OF[h]][0]
    est_name = t["metric"]["estimator"]
    g_mod = {est_name: float(t["metric"]["g_mod"]),
             **{k: float(v) for k, v in t["metric"]["report_only"]["g_mod"].items()}}
    span = P.HORIZONS[h]
    need = L.MIN_MONTHS if h == 12 else L.MIN_MONTHS24
    cols, per = rows(ms, mapping, pf, heads, span, fallback_changes)
    gate_obs = all(per[k]["n_months"] >= need for k in L.PRIMARY)
    gate_model = bool(ms["informative_model_side_pass"])
    out = {"test_id": t["test_id"], "role": t["role"], "estimator": est_name,
           "g_mod": g_mod, "piles": per, "informative": gate_obs and gate_model}
    if not out["informative"]:
        why = ("model-side gate failed" if not gate_model else
               f"fewer than {need} months with a verified pile value and a nest dD at a "
               "primary pile")
        out.update({"verdict": "not_informative", "why": why,
                    "wording": WORDING["not_informative"].format(why=why)})
        return out
    est = P.estimate_all(cols["y"], cols["x"], cols["z"], cols["m"], np.asarray(cols["g"]),
                         rng=np.random.default_rng(L.BOOT_SEED))
    v = P.verdicts(est, g_mod)
    main_v = v[est_name]
    out.update({"estimate": est[est_name], **main_v,
                "first_stage_f": {"f_harmonic": est["f_harmonic"],
                                  "f_model": est["f_model"]}})
    if main_v["rho"] is not None and main_v["rho_ci"] is not None:
        out["wording"] = WORDING[main_v["verdict"]].format(
            rho=main_v["rho"], lo=main_v["rho_ci"][0], hi=main_v["rho_ci"][1])
    else:
        out["wording"] = ("compaction per metre of head change at the 2 Yunlin pile cells: "
                          f"{main_v['verdict'].replace('_', ' ')} (estimate or interval not "
                          "finite; N3G)")
    out["report_only"] = {k: {"estimate": est[k], **v[k]} for k in P.ALL_LINES
                          if k != est_name}
    if t["role"] != "gating":
        out["label"] = "report-only (the other horizon); C31 is moved only by the gating test"
    return out


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    from common import guard
    from harness import prereg3

    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    out = Path(a.out) / "report.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the report is written once")
    h = int(a.horizon)
    nb, nl = (L.NOT_BEFORE, L.NO_LATER_THAN) if h == 12 else (L.NOT_BEFORE24,
                                                              L.NO_LATER_THAN24)
    now = L._now()
    if now < nb:
        raise ScoringRefused(f"not before {nb:%Y-%m-%d}")
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    doc, _ = prereg3.load_prereg(P.N3G_ID, **kw)
    rep = {"prereg_id": P.N3G_ID, "horizon_months": h, "scorer_version": SCORER_VERSION,
           "scored_at": now.isoformat(), "roles": doc["roles"]}
    ok_p, why_p = S.product_grants(a.piles_product)
    ok_h, why_h = S.product_grants(a.heads_product)
    rep["products"] = {"piles": why_p, "heads": why_h}
    if now > nl or not (ok_p and ok_h):
        why = ("after the no_later_than date" if now > nl else
               "no product grants prospective_scorer")
        if now <= nl and not getattr(a, "final", False):
            raise ScoringRefused(f"{why}: build the products first, or pass --final to "
                                 "write the not-evaluable record (nothing read, no look)")
        rep.update({"verdict": "not_informative", "why": why})
        S._write_once(out, rep)
        return rep
    order = n3_looks_done(pdir, h)
    rep["order"] = order
    if not order["ok"]:
        raise ScoringRefused(f"session order: N3's step-(11) score looks first ({order})")
    t = TEST_OF[h]
    rep["post_registration_looks"] = prereg3.assert_can_score(
        P.N3G_ID, P.WP, test_id=t, **kw)["post_registration_looks"]
    files = doc["binds"]["model_side"]
    from common.paths import from_repo

    mapping, ms = (json.loads(from_repo(files[n]).read_text(encoding="utf-8"))
                   for n in ("mapping.json", "model_side.json"))
    piles = guard.open_product(a.piles_product, "prospective_scorer")
    raw = piles.read_table(S.PILE_TABLE)
    pf = S.pile_frame(raw)
    ids = sorted({w["sid"] for k in L.PRIMARY for w in mapping["piles"][k]["nest"]})
    hp = guard.open_product(a.heads_product, "prospective_scorer")
    heads = S.heads_series(hp.read_table(a.heads_table), ids)
    fb = None
    if S.WELL_TABLE in piles.outputs and a.stations:
        fb = S.fallback_changes(piles.read_table(S.WELL_TABLE), mapping, Path(a.stations))
    rep["T_conversion"] = P.jsonable(conversion(doc, ms, mapping, pf, heads, h, fb))
    S._write_once(out, rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--horizon", choices=("12", "24"), default="12")
    s.add_argument("--piles-product", type=Path, required=True)
    s.add_argument("--heads-product", type=Path, required=True)
    s.add_argument("--heads-table", default="heads_monthly.parquet")
    s.add_argument("--stations", default=None)
    s.add_argument("--final", action="store_true",
                   help="write a not-informative record when no product grants "
                        "prospective_scorer by the deadline")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = run_score(a)
    tc = rep.get("T_conversion") or {}
    print(json.dumps({"verdict": tc.get("verdict", rep.get("verdict")),
                      "wording": tc.get("wording", rep.get("why"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
