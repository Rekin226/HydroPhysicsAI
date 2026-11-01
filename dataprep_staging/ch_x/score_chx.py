"""Score I1.CHX.v1 once: R1c's Changhua pumping response direction at the 23 never-calibrated
external Changhua screens (primary), the Yunlin-only placebo (report only) and, when CH-J ran,
CH-J's external gates (secondary) - one session, one report.

Order (each step refuses before the next; no head is opened before step 4):

0. the declaration is registered and intact; this file and every frozen code file match the
   declared sha256 (``--verify-registration`` runs only this step);
1. the hindcast panel matches chx_panel.json, was built at the declared code commit from the
   declared theta, and carries the declared 23 screens (rows sha256);
2. the informative gate on R1c's transformed contrast (panel only): if it fails, the report
   says 'not_informative' and NO head is read; then G0 (g0.json) was run on this panel with
   the registered code: exact plumbing passed and the gate says decidable; if G0 says 'not
   decidable' the report is written with that verdict and NO head is read (no scoring look
   is logged);
3. ``prereg3.assert_can_score`` (logs the scoring look);
4. heads: L1_clean v3, use design_time_evidence, window_end 2022-12, flags 0-1, the 23
   screens, 2013-01..2022-12 (``chx_io.l1_monthly``);
5. primary (``chx_lib.score_direction``, form ``chx_const.CHX_PRIMARY_FORM``), placebo
   (Yunlin-only contrast), CH-J gates if their panel is given - gating only when the CH-J
   engineering record verifies (``chj_provenance``), report-only otherwise; report.json
   written once (mode 'x').

    python dataprep_staging/ch_x/score_chx.py --verify-registration
    python dataprep_staging/ch_x/score_chx.py score --panel <chx_panel.npz> --g0 <g0.json> \\
        [--chj-panel <chj_panel.npz> --chj-metrics <chj_metrics.json> \\
         --chj-engineering <chj_engineering.json>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402

SCORER_VERSION = "1.0.0"
REPORT = f"results/dataprep/reports/{K.CHX_ID}/report.json"
FROZEN = (f"{K.X}/chx_const.py", f"{K.X}/chx_lib.py", f"{K.X}/chx_io.py",
          f"{K.X}/hindcast.py", f"{K.X}/g0_chx.py", f"{K.X}/score_chx.py",
          f"{K.X}/run_chx.sh", f"{K.X}/chj_forcing.py", K.SCORER,
          "dataprep_staging/n2c_reading/synth_n2c.py",
          "dataprep_staging/n1_policy/policy_lib.py")
WORDING = {
    "direction": "R1c's Changhua pumping response has the right direction at 23 "
                 "never-calibrated Changhua screens (design-time spatial holdout; verdict "
                 "{verdict}; slope {beta:.2f} [{lo:.2f}, {hi:.2f}], {form} fixed effects)",
    "not_supported": "R1c's Changhua pumping response direction is not supported at 23 "
                     "never-calibrated Changhua screens (design-time spatial holdout)",
    "not_informative": "R1c's Changhua contrast at the 23 never-calibrated screens is not "
                       "informative (the pinned informative gate on the transformed "
                       "contrast); no direction reading; no head was read",
    "not_decidable": "the direction test at the 23 never-calibrated Changhua screens is not "
                     "decidable (G0: power {power:.2f}, confounded false pass {false:.2f}); "
                     "no head was read",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def frozen_code() -> dict:
    return {p: IO.sha_or_missing(HERE.parents[1] / p) for p in FROZEN}


# ----------------------------------------------------------------------------- step 0
def verify_registration(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    from harness import prereg3

    kw = {}
    if prereg_dir is not None:
        kw.update(prereg_dir=prereg_dir, rows_dir=rows_dir)
    doc, sha = prereg3.load_prereg(K.CHX_ID, **kw)
    now = frozen_code()
    bad = {p: (doc["frozen_code"].get(p), h) for p, h in now.items()
           if doc["frozen_code"].get(p) != h}
    if bad:
        raise SystemExit(f"REFUSED: frozen code differs from the declaration: {bad}")
    return doc, sha


# ----------------------------------------------------------------------------- steps 1-2
def check_panel(panel_path: Path, doc: dict) -> dict:
    summ = json.loads((panel_path.parent / "chx_panel.json").read_text())
    probs = []
    if IO.sha256(panel_path) != summ["panel_sha256"]:
        probs.append("panel sha256 differs from chx_panel.json")
    if summ.get("code_head") != doc["recipe"]["worktree_head"]:
        probs.append(f"panel built at {summ.get('code_head')}, declared "
                     f"{doc['recipe']['worktree_head']}")
    if summ.get("theta_sha256") != doc["recipe"]["theta_sha256"]:
        probs.append("theta sha256 differs from the declaration")
    if summ.get("rows_sha256_changhua") != doc["screens"]["rows_sha256"]:
        probs.append("the panel's Changhua rows differ from the declared 23 screens")
    z = np.load(panel_path, allow_pickle=False)
    sid = sorted(np.asarray(z["sid_ch"]).astype(str))
    if sid != sorted(doc["screens"]["ids"]):
        probs.append("panel screens != declared ids")
    if [str(m) for m in z["months"]] != L.month_list(*K.CHX_WINDOW):
        probs.append("panel months are not the declared window")
    if probs:
        raise SystemExit("REFUSED (step 1): " + "; ".join(probs))
    return {k: z[k] for k in z.files} | {"_summary": summ}


def check_g0(g0_path: Path, panel_path: Path, doc: dict) -> dict:
    g = json.loads(g0_path.read_text())
    probs = []
    if g.get("smoke"):
        probs.append("g0 is a smoke run")
    if g.get("panel_sha256") != IO.sha256(panel_path):
        probs.append("g0 ran on another panel")
    for p, h in (g.get("code_sha256") or {}).items():
        if doc["frozen_code"].get(f"{K.X}/{p}") != h:
            probs.append(f"g0 ran with {p} other than declared")
    if not g.get("plumbing", {}).get("ok"):
        probs.append("g0 exact plumbing failed (engineering)")
    if g.get("design", {}).get("replicates_per_cell") != K.G0_REPLICATES:
        probs.append("g0 replicates differ from the declaration")
    if g.get("design", {}).get("form") != K.CHX_PRIMARY_FORM:
        probs.append("g0 ran another primary form")
    if probs:
        raise SystemExit("REFUSED (step 2): " + "; ".join(probs))
    return g


# ----------------------------------------------------------------------------- step 5
def score_panel(P: dict, obs: np.ndarray, flag: np.ndarray) -> dict:
    """Primary and placebo from the panel and the observed (W, T) arrays (no file access)."""
    months = [str(m) for m in P["months"]]
    sid = np.asarray(P["sid_ch"]).astype(str)
    blocks = dict(zip(sid, np.asarray(P["block_ch"]).astype(str), strict=True))
    inf = informative(P)
    rows = L.eligible(L.long_panel(sid, months, obs, P["B_ch"], P["c_ch"], flag))
    prim = L.score_direction(rows, blocks, inf)
    prim["dropped_wells_lt_min_months"] = rows["dropped_wells"]
    inf_pl = L.informative_from_contrast(np.asarray(P["c_placebo_ch"], float), sid, blocks,
                                         months)
    rows_pl = L.eligible(L.long_panel(sid, months, obs, P["C_ch"], P["c_placebo_ch"], flag))
    pl = L.score_direction(rows_pl, blocks, inf_pl)
    return {"primary": prim, "placebo_report_only": pl}


def informative(P: dict) -> dict:
    """The primary's informative gate from the panel alone (no head)."""
    months = [str(m) for m in P["months"]]
    sid = np.asarray(P["sid_ch"]).astype(str)
    blocks = dict(zip(sid, np.asarray(P["block_ch"]).astype(str), strict=True))
    return L.informative_from_contrast(np.asarray(P["c_ch"], float), sid, blocks, months)


#: the CH-J engineering record, written BEFORE any CH-J fit (chj_engineering.json)
ENGINEERING_KEYS = ("hook_commit", "hook_sha256", "forcing_version", "forcing_sha256",
                    "pregate", "frozen_at", "producer_sha256")
#: the CH-J panel producer's summary beside chj_panel.npz (chj_panel.json), written after it
PANEL_SUMMARY_KEYS = ("engineering_sha256", "producer_sha256", "chj_panel_sha256",
                      "chj_metrics_sha256", "created")


def _utc(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def chj_provenance(doc: dict, eng_path: Path | None, panel: Path, metrics: Path,
                   seeds) -> dict:
    """May CH-J's gates gate? Only if (1) the engineering record ``eng_path`` (keys
    ENGINEERING_KEYS) exists, was frozen within CHJ_ENGINEERING_DEADLINE_DAYS of the
    registration's ``created`` and records a passed freeze pre-gate within
    CHJ_FREEZE_REPRO_TOL_M; (2) the producer's ``chj_panel.json`` beside ``panel`` (keys
    PANEL_SUMMARY_KEYS) names that record (sha256) and its producer, this very panel and
    metrics (sha256), and was created after the freeze; (3) the metrics hold exactly the
    declared seeds. Otherwise CH-J is report-only (the declared fallback)."""
    probs = []
    if sorted(str(s) for s in seeds) != sorted(str(s) for s in K.CHJ_SEEDS):
        probs.append(f"seeds {sorted(map(str, seeds))} are not exactly {list(K.CHJ_SEEDS)}")
    eng = summ = None
    if eng_path is None or not Path(eng_path).is_file():
        probs.append("no CH-J engineering record")
    else:
        eng = json.loads(Path(eng_path).read_text())
        miss = [k for k in ENGINEERING_KEYS if k not in eng]
        if miss:
            probs.append(f"engineering record lacks {miss}")
            eng = None
    if eng is not None:
        limit = _utc(doc["created"]) + dt.timedelta(days=K.CHJ_ENGINEERING_DEADLINE_DAYS)
        if _utc(eng["frozen_at"]) > limit:
            probs.append(f"CH-J frozen at {eng['frozen_at']}, after the "
                         f"{K.CHJ_ENGINEERING_DEADLINE_DAYS}-day deadline")
        pg = eng["pregate"] or {}
        d = pg.get("max_abs_diff_m")
        if not (pg.get("passed") and d is not None and float(d) <= K.CHJ_FREEZE_REPRO_TOL_M):
            probs.append("the freeze-hook pre-gate did not pass")
        sp = Path(panel).parent / "chj_panel.json"
        summ = json.loads(sp.read_text()) if sp.is_file() else None
        if summ is None or any(k not in summ for k in PANEL_SUMMARY_KEYS):
            probs.append("no complete chj_panel.json beside the CH-J panel")
            summ = None
    if summ is not None:
        checks = {"engineering record": (summ["engineering_sha256"], IO.sha256(eng_path)),
                  "producer": (summ["producer_sha256"], eng["producer_sha256"]),
                  "chj_panel.npz": (summ["chj_panel_sha256"], IO.sha256(panel)),
                  "chj_metrics.json": (summ["chj_metrics_sha256"], IO.sha256(metrics))}
        probs += [f"{name} differs from chj_panel.json" for name, (x, y) in checks.items()
                  if x != y]
        if _utc(summ["created"]) < _utc(eng["frozen_at"]):
            probs.append("the CH-J panel predates the engineering freeze")
    return {"gating": not probs, "problems": probs, "engineering_record": eng,
            "panel_summary": summ}


def chj_block(P: dict, obs_ch, obs_yl, chj_panel: Path | None, chj_metrics: Path | None,
              doc: dict | None = None, chj_engineering: Path | None = None,
              direction_fn=L.score_direction) -> dict:
    """CH-J's gates (secondary). ``chj_panel`` keys: chjA_ch_<seed>, chjB_ch_<seed>,
    chjA_yl_<seed>, q0_ch (window heads at the externals; chjB = CH-J with Changhua at its
    2012 pattern); ``chj_metrics``: {"seeds": {seed: {"noharm": {...}, "q0": {...},
    "forward": {...}, "forward_q0": {...}, "sw_fraction_changhua": f}}}. The gates GATE only
    when ``chj_provenance`` passes; otherwise every number is reported and nothing gates."""
    if chj_panel is None:
        return {"status": "not_run", "reading": "CH-J engineering not delivered: the declared "
                "fallback applies (primary scored alone; CH-J external gates report-only "
                "when run later, never gating)"}
    z = np.load(chj_panel, allow_pickle=False)
    met = json.loads(Path(chj_metrics).read_text())
    months = [str(m) for m in P["months"]]
    sid = np.asarray(P["sid_ch"]).astype(str)
    blocks = dict(zip(sid, np.asarray(P["block_ch"]).astype(str), strict=True))
    prov = chj_provenance(doc or {"created": "1970-01-01T00:00:00+00:00"}, chj_engineering,
                          chj_panel, chj_metrics, met["seeds"].keys())
    out = {"status": "run" if prov["gating"] else "report_only", "provenance": prov,
           "panel_sha256": IO.sha256(chj_panel), "metrics_sha256": IO.sha256(chj_metrics),
           "seeds": {}}
    for seed, s in met["seeds"].items():
        A, B = z[f"chjA_ch_{seed}"], z[f"chjB_ch_{seed}"]
        inf = L.informative_from_contrast(A - B, sid, blocks, months)
        d = direction_fn(L.eligible(L.long_panel(sid, months, obs_ch, B, A - B)), blocks, inf)
        ext = L.chj_external_gates(A, obs_ch, z[f"chjA_yl_{seed}"], obs_yl, months,
                                   z["q0_ch"], d)
        sw = L.chj_sw_outcome(float(s["sw_fraction_changhua"]))
        nh = L.chj_noharm_gates(s["noharm"], s["q0"])
        fw = L.chj_forward_gates(s["forward"], s["forward_q0"])
        out["seeds"][seed] = {"sw_outcome": sw, "noharm": nh, "forward": fw, "external": ext,
                              "passed": bool(sw == "off_bound" and nh["passed"]
                                             and fw["passed"] and ext["passed"])}
    out["passed_all_seeds"] = bool(prov["gating"] and out["seeds"]
                                   and all(v["passed"] for v in out["seeds"].values()))
    out["report_only_all_seeds_pass"] = bool(out["seeds"] and all(
        v["passed"] for v in out["seeds"].values()))
    out["co_located_recharge_rejected"] = bool(prov["gating"] and any(
        v["sw_outcome"] == "co_located_recharge_rejected" for v in out["seeds"].values()))
    return out


def wording(prim: dict) -> str:
    v = prim["verdict"]
    if prim["direction"]:
        b = prim["beta_primary"]
        return WORDING["direction"].format(verdict=v, beta=b["beta"], lo=b["ci90"][0],
                                           hi=b["ci90"][1], form=prim["form"])
    return WORDING["not_informative" if v == "not_informative" else "not_supported"]


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    out = IO.rp(REPORT)
    if out.exists():
        raise SystemExit(f"{REPORT} exists; I1.CHX.v1 is scored once")
    doc, decl_sha = verify_registration(prereg_dir, rows_dir)
    P = check_panel(a.panel, doc)
    rep = {"prereg_id": K.CHX_ID, "prereg_sha256": decl_sha, "scorer_version": SCORER_VERSION,
           "created": _now(), "panel": str(a.panel), "panel_sha256": IO.sha256(a.panel),
           "form": K.CHX_PRIMARY_FORM}
    inf = informative(P)
    if not inf["passes"]:
        rep.update({"verdict": "not_informative", "direction": False, "heads_read": False,
                    "informative_gate": inf, "wording": WORDING["not_informative"]})
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as f:
            json.dump(rep, f, indent=1, default=float)
        print(f"not informative (panel only); no head read -> {REPORT}")
        return rep
    g0 = check_g0(a.g0, a.panel, doc)
    rep.update({"g0": str(a.g0), "g0_sha256": IO.sha256(a.g0), "g0_gate": g0["gate"]})
    if not g0["gate"]["decidable"]:
        rep.update({"verdict": "not_decidable", "heads_read": False,
                    "wording": WORDING["not_decidable"].format(
                        power=g0["gate"]["power_k1_rho083"],
                        false=g0["gate"]["false_pass_k0_max_rho"])})
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as f:
            json.dump(rep, f, indent=1, default=float)
        print(f"not decidable (G0); no head read -> {REPORT}")
        return rep
    from harness import prereg3

    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    acc = prereg3.assert_can_score(K.CHX_ID, K.WP, test_id=K.CHX_TEST, **kw)
    months = [str(m) for m in P["months"]]
    sid = list(np.asarray(P["sid_ch"]).astype(str))
    obs, flag, notes = IO.l1_monthly(sid, months)
    res = score_panel(P, obs, flag)
    obs_yl = None
    if a.chj_panel is not None:
        acc2 = prereg3.assert_can_score(K.CHX_ID, K.WP, test_id=K.CHJ_TEST, **kw)
        rep["chj_post_registration_looks"] = acc2["post_registration_looks"]
        obs_yl, _, _ = IO.l1_monthly(list(np.asarray(P["sid_yl"]).astype(str)), months)
    res["chj_secondary"] = chj_block(P, obs, obs_yl, a.chj_panel, a.chj_metrics, doc,
                                     getattr(a, "chj_engineering", None))
    rep.update({"heads_read": True, "guard_notes": notes,
                "post_registration_looks": acc["post_registration_looks"],
                "unverifiable_prior_looks": acc["unverifiable_prior_looks"][:20],
                "verdict": res["primary"]["verdict"],
                "direction": res["primary"]["direction"],
                "wording": wording(res["primary"]), **res})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"{K.CHX_ID}: {rep['verdict']} -> {REPORT}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-registration", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--panel", type=Path, required=True)
    s.add_argument("--g0", type=Path, required=True)
    s.add_argument("--chj-panel", type=Path, default=None)
    s.add_argument("--chj-metrics", type=Path, default=None)
    s.add_argument("--chj-engineering", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.verify_registration:
        _, sha = verify_registration()
        print(f"{K.CHX_ID} registered and intact (sha256 {sha[:16]}); frozen code matches")
        return 0
    if a.cmd != "score":
        ap.error("use --verify-registration or score")
    if (a.chj_panel is None) != (a.chj_metrics is None):
        ap.error("--chj-panel and --chj-metrics go together")
    run_score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
