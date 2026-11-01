"""Score I1.CH_backcast.v1 once: the direction of R1c's Changhua pumping response in
2010-01..2011-12, months no model has fitted (design-time; high-frequency band previously
read by X5EV).

Order (no head is opened before step 4):

0. the registration is intact and every frozen code file matches;
1. the availability look passed its stop rule and S0 (code-path identity and the declared
   forcing-swap gate) passed (both frozen in the registration by sha256);
2. heads/L1_pre2012 v1: its MANIFEST allows design_time_evidence (read allowed_uses before
   gating), it was built from this registration (parameters.registration.sha256) for the
   registered wells;
3. the backcast panel (hindcast.py backcast) matches bc_panel.json and holds only registered
   wells; the report-only synthetic power is computed from the panel (model only); then
   ``prereg3.assert_can_score`` (logs the scoring look);
4. heads: L1_pre2012 v1, use design_time_evidence, window_end 2011-12, flags 0-1;
5. 'not_evaluable' if fewer than 10 wells keep >= 18 scored months; otherwise
   ``chx_lib.score_backcast`` (well x calendar-month and month-of-sample effects, pinned FE
   slope, 90 % 5 km block interval; pass iff the informative gate on the transformed
   contrast passes and the interval lies above 0). report.json once.

    python dataprep_staging/ch_x/score_bc.py --verify-registration
    python dataprep_staging/ch_x/score_bc.py score --panel <bc_panel.npz>
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
REPORT = f"results/dataprep/reports/{K.BC_ID}/report.json"
AVAIL = f"{K.BC_DIR}/avail/avail.json"
S0 = f"{K.BC_DIR}/s0/s0.json"
FROZEN = (f"{K.X}/chx_const.py", f"{K.X}/chx_lib.py", f"{K.X}/chx_io.py",
          f"{K.X}/bc_lib.py", f"{K.X}/hindcast.py", f"{K.X}/avail_l1pre.py",
          f"{K.X}/build_l1pre.py", f"{K.X}/score_bc.py", f"{K.X}/run_backcast.sh",
          K.SCORER, "dataprep_staging/h1_heads_qc/screens.py",
          "dataprep_staging/h1_heads_qc/config.py", K.G0_PARAMS_FROM)
WORDING = {
    "direction_supported": "direction of R1c's Changhua pumping response supported in years "
                           "never used to fit any model (2010-11, design-time; slope "
                           "{beta:.2f} [{lo:.2f}, {hi:.2f}], well x calendar-month and "
                           "month-of-sample effects)",
    "not_supported": "direction of R1c's Changhua pumping response not supported in 2010-11 "
                     "(design-time)",
    "not_informative": "R1c's Changhua contrast in 2010-11 is not informative (the pinned "
                       "gate on the transformed contrast); no reading",
    "not_evaluable": "the 2010-11 Changhua backcast is not evaluable (fewer than {n} wells "
                     "with {m} scored months after QC); U-F closed as not evaluable",
}


def frozen_code() -> dict:
    return {p: IO.sha_or_missing(HERE.parents[1] / p) for p in FROZEN}


def verify_registration(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    from harness import prereg3

    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    doc, sha = prereg3.load_prereg(K.BC_ID, **kw)
    bad = {p: (doc["frozen_code"].get(p), h) for p, h in frozen_code().items()
           if doc["frozen_code"].get(p) != h}
    if bad:
        raise SystemExit(f"REFUSED: frozen code differs from the registration: {bad}")
    return doc, sha


def registered_ids(doc: dict) -> list[str]:
    return sorted(doc["tests"][0]["external_units"][K.BC_UNIT]["ids"])


def check_pregates(doc: dict) -> None:
    probs = []
    for key, path in (("availability", AVAIL), ("s0", S0)):
        p = IO.rp(path)
        if IO.sha_or_missing(p) != doc["pregates"][key]["sha256"]:
            probs.append(f"{path} differs from the registration")
    if not doc["pregates"]["availability"]["passed"] or not doc["pregates"]["s0"]["passed"]:
        probs.append("a pre-gate did not pass")
    if probs:
        raise SystemExit("REFUSED (step 1): " + "; ".join(probs))


def check_product(doc: dict, decl_sha: str, product: str = K.L1PRE_V1) -> dict:
    from common.manifest import load_manifest

    m, _ = load_manifest(IO.rp(product))
    probs = []
    if "design_time_evidence" not in m.get("allowed_uses", []):
        probs.append("L1_pre2012 does not allow design_time_evidence")
    reg = (m.get("parameters") or {}).get("registration") or {}
    if reg.get("id") != K.BC_ID or reg.get("sha256") != decl_sha:
        probs.append("L1_pre2012 was not built from this registration")
    if sorted((m.get("parameters") or {}).get("wells") or []) != registered_ids(doc):
        probs.append("L1_pre2012's wells differ from the registered unit")
    if probs:
        raise SystemExit("REFUSED (step 2): " + "; ".join(probs))
    return {"allowed_uses": m["allowed_uses"], "gate": m.get("gate")}


def check_panel(panel: Path, doc: dict) -> dict:
    summ = json.loads((panel.parent / "bc_panel.json").read_text())
    probs = []
    if IO.sha256(panel) != summ["panel_sha256"]:
        probs.append("panel sha256 differs from bc_panel.json")
    if summ.get("code_head") != doc["recipe"]["worktree_head"]:
        probs.append("panel not built at the declared commit")
    if summ.get("theta_sha256") != doc["recipe"]["theta_sha256"]:
        probs.append("theta differs")
    z = np.load(panel, allow_pickle=False)
    if not set(np.asarray(z["sid"]).astype(str)) <= set(registered_ids(doc)):
        probs.append("panel wells outside the registered unit")
    if [str(m) for m in z["months"]] != L.month_list(*K.BC_WINDOW):
        probs.append("panel months are not 2010-01..2011-12")
    if probs:
        raise SystemExit("REFUSED (step 3): " + "; ".join(probs))
    return {k: z[k] for k in z.files}


def score_panel(P: dict, obs: np.ndarray, flag: np.ndarray) -> dict:
    """The verdict from the panel and observed (W, T) arrays (no file access)."""
    months = [str(m) for m in P["months"]]
    sid = np.asarray(P["sid"]).astype(str)
    blocks = dict(zip(sid, np.asarray(P["block"]).astype(str), strict=True))
    rows = L.eligible(L.long_panel(sid, months, obs, P["B"], P["c"], flag),
                      min_months=K.BC_MIN_MONTHS)
    n_w = len(set(rows["well"]))
    if n_w < K.BC_MIN_WELLS:
        return {"verdict": "not_evaluable", "passed": False, "n_wells_scored": n_w,
                "dropped_wells": rows["dropped_wells"],
                "wording": WORDING["not_evaluable"].format(n=K.BC_MIN_WELLS,
                                                           m=K.BC_MIN_MONTHS)}
    keep = np.isin(sid, sorted(set(rows["well"])))
    inf = L.informative_from_contrast(np.asarray(P["c"], float)[keep], sid[keep],
                                      blocks, months, form=K.BC_PRIMARY_FORM)
    r = L.score_backcast(rows, blocks, inf)
    r["dropped_wells"] = rows["dropped_wells"]
    b = r["beta_primary"]
    r["wording"] = (WORDING["direction_supported"].format(beta=b["beta"], lo=b["ci90"][0],
                                                          hi=b["ci90"][1])
                    if r["verdict"] == "direction_supported" else WORDING[r["verdict"]])
    return r


def power(P: dict) -> dict:
    """Report only: ``chx_lib.bc_power`` on the panel (model output and N2C's declared
    parameters; no head)."""
    sys.path.insert(0, str(HERE.parent / "n2c_reading"))
    import synth_n2c as SN

    par = {k: float(SN.PARAMS[k]["value"]) for k in ("sigma_tot_m", "block_share", "phi",
                                                      "kappa_m")}
    months = [str(m) for m in P["months"]]
    sid = np.asarray(P["sid"]).astype(str)
    blocks = dict(zip(sid, np.asarray(P["block"]).astype(str), strict=True))
    inf = L.informative_from_contrast(np.asarray(P["c"], float), sid, blocks, months,
                                      form=K.BC_PRIMARY_FORM)
    if not inf["passes"]:
        return {"computed": False, "reason": "contrast not informative at the panel wells"}
    return {"computed": True, **L.bc_power(P, par, inf)}


def run_score(a, prereg_dir=None, rows_dir=None, power_fn=power) -> dict:
    out = IO.rp(REPORT)
    if out.exists():
        raise SystemExit(f"{REPORT} exists; I1.CH_backcast.v1 is scored once")
    doc, sha = verify_registration(prereg_dir, rows_dir)
    check_pregates(doc)
    prod = check_product(doc, sha)
    P = check_panel(a.panel, doc)
    pw = power_fn(P)
    from harness import prereg3

    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    acc = prereg3.assert_can_score(K.BC_ID, K.WP, test_id=K.BC_TEST, **kw)
    months = [str(m) for m in P["months"]]
    obs, flag, notes = IO.l1_monthly(list(np.asarray(P["sid"]).astype(str)), months,
                                     product=K.L1PRE_V1, use="design_time_evidence",
                                     window_end=K.BC_WINDOW[1])
    res = score_panel(P, obs, flag)
    rep = {"prereg_id": K.BC_ID, "prereg_sha256": sha, "scorer_version": SCORER_VERSION,
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "panel": str(a.panel), "panel_sha256": IO.sha256(a.panel), "product": prod,
           "guard_notes": notes, "post_registration_looks": acc["post_registration_looks"],
           "unverifiable_prior_looks": acc["unverifiable_prior_looks"][:20],
           "report_only_synthetic_power": pw, **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"{K.BC_ID}: {rep['verdict']} -> {REPORT}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-registration", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--panel", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.verify_registration:
        _, sha = verify_registration()
        print(f"{K.BC_ID} registered and intact (sha256 {sha[:16]})")
        return 0
    if a.cmd != "score":
        ap.error("use --verify-registration or score")
    run_score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
