"""I1.X5SC.gate0 evaluator: the gate-0 verdict of each slow form from the synthetic fits (CPU,
about a minute). Scored once.

    results/twin_runs/par32/py.sh dataprep_staging/x5_slow/evaluate_e2.py

Arms are ``<runs>/<truth>_<form>_r<year>/`` (run_e2.sh), each holding calibrate_coupled's
``vep_zonal_leveling.json`` and jobmeta's ``job.json``; ``<runs>/attempts.jsonl`` is the
attempt ledger. Every number is synthetic: the fitted column and the truth column are both
driven by the generator's truth heads (calibrate_coupled's own nudged hindcast, months
<= 2021-12), so the test isolates identifiability from head-forecast error.
``rings_independent_r2`` (real rings vs a synthetic fit) is dropped on load and never
reported.

Gates (e2common.SPEC["gates"], frozen in the declaration before any arm exists):
  G0-A1  truth A: next-year distal background, fit/truth in [0.67, 1.5], both origins
  G0-A2  truth A, aquitard form: distal Ska/tau_a / 0.006 in [0.5, 2], both origins
  G0-A3  truth A: distal slow parameters not at a bound (1 % of the bound width)
  G0-N   truth N: |next-year distal background| < 0.5 cm/yr, both origins
  eligible for I1.X5SC = A1 and N and A3 and (aquitard) A2
  G0-V   claim gate, aquitard form: truth V fitted Ska/tau_a < 0.5 x 0.006, both origins
An arm that failed, is non-conforming (argv, data dir, panel hash, spec, code, ledger) or
ran while the code changed makes every gate that needs it "not evaluable", which counts as
not passed.

Refusals (nothing is written): the declaration is not registered or its frozen hashes do not
match (spec, generator manifest, code tree, runs dir); the report exists or the ledger
records an evaluation; the runner still holds its lock; any declared arm has no ended
attempt or an open one; an arm has more than one completed attempt. The report path is
fixed: ``<runs>/e2_report.json`` (the declaration's ``frozen.report``); it is written once
and the ledger records its sha256.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as C  # noqa: E402
import jobmeta as J  # noqa: E402

G = C.SPEC["gates"]
AQ_FORMS = ("aq",)
SLOW_KEYS = {"aq": ("log_ska", "log_tau_a", "h_a0"), "vt": ("log_tau",)}
REPORT_NAME = Path(C.REPORT).name
LOCK_NAME = ".runner.lock"


def _flag(flags: list[str], name: str):
    return flags[flags.index(name) + 1] if name in flags else None


def _time(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------------------
# conformity: the arm ran what the declaration froze
# ---------------------------------------------------------------------------------------
def conformity(p: dict, job: dict | None, truth: str, form: str, origin: str,
               runs: Path, synth_manifest: dict, decl: dict, status: dict) -> list[str]:
    """Reasons the arm does not conform (empty = conforms)."""
    from hydrophysics.twin.forward import check_column_creep

    bad = []
    try:
        check_column_creep(p, label="")
    except ValueError as e:
        bad.append(f"forward refuses the column: {e}")
    flags = C.form_flags(form)
    want = {"config": "zonal", "target": "leveling", "hindcast_gain": 1.0,
            "hindcast_every": 1, "update_method": "innovation", "apex_hold": "calibrated",
            "hpc0_guard_days": 365.0, "fold_seed": 2027,
            "withhold_surveys": C.SPEC["withhold"].format(year=origin[:4]),
            "creep": "aquitard" if form in AQ_FORMS else "vep"}
    if form in AQ_FORMS:
        want.update(creep_zones=_flag(flags, "--creep-zones"),
                    aquitard_tau_min_years=float(_flag(flags, "--aquitard-tau-min-years")),
                    aquitard_ha0_min=float(_flag(flags, "--aquitard-ha0-min")),
                    tau_max_years=None)
    else:
        z, d = _flag(flags, "--tau-min-days-zone").split("=")
        want.update(tau_min_days_zone={z: float(d)},
                    tau_max_years=float(_flag(flags, "--tau-max-years")))
    for k, v in want.items():
        got = p.get(k)
        if isinstance(v, float) and got is not None:
            ok = isinstance(got, (int, float)) and math.isclose(float(got), v)
        else:
            ok = got == v
        if not ok:
            bad.append(f"{k}: recorded {got!r}, declared {v!r}")
    if not str(p.get("theta", "")).endswith(C.SPEC["theta"]):
        bad.append(f"theta {p.get('theta')!r} is not {C.SPEC['theta']}")
    if p.get("zone_blend_km") or p.get("banded") or len(p.get("zonal", [])) != 3:
        bad.append("not Q0's sharp three-zone column")
    if job is None:
        bad.append("no job.json")
        return bad
    arm = C.arm_name(truth, form, origin)
    tv = synth_manifest["truths"].get(truth)
    if job.get("rc") != 0:
        bad.append(f"rc {job.get('rc')}")
    if job.get("truth") != truth or job.get("form") != form or job.get("origin") != origin:
        bad.append("job.json labels disagree with the arm name")
    if tv is None or job.get("panel_sha256") != tv["panel_sha256"]:
        bad.append("panel hash differs from the generator's truth panel")
    if tv is None or job.get("cmd") != C.fit_args(form, origin, tv["data_dir"],
                                                  str(runs.resolve() / arm)):
        bad.append("the recorded argv is not the spec's")
    if tv is None or Path(job.get("data_dir", "")).resolve() != Path(tv["data_dir"]).resolve() \
            or Path(job.get("env_HYDROMIND_GW_DATA") or "/").resolve() \
            != Path(tv["data_dir"]).resolve() or job.get("guard") != "passed":
        bad.append("the data-dir guard did not record the synthetic dir for --data and "
                   "HYDROMIND_GW_DATA")
    if job.get("spec_sha256") != C.spec_sha256():
        bad.append("spec changed since the job started")
    if job.get("code_changed_during_job"):
        bad.append("the code changed while the job ran (code freeze)")
    if (job.get("code_start") or {}).get("tree_sha256") != decl["frozen"]["code"]["tree_sha256"]:
        bad.append("the arm ran on code other than the declared frozen tree")
    if _time(job.get("start", "1970-01-01T00:00:00+00:00")) < _time(decl["created"]):
        bad.append("the arm started before the declaration was registered")
    if status["completed"] != [job.get("attempt")]:
        bad.append(f"ledger: completed attempts {status['completed']}, job.json attempt "
                   f"{job.get('attempt')}")
    return bad


# ---------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------
class World:
    """Truth heads, grid, zones and the truths' columns and synthetic panels."""

    def __init__(self, synth_dir: Path):
        from hydrophysics.twin.leveling import load_panel

        self.synth_dir = synth_dir
        self.man = J.check_synth(synth_dir)
        H = C.truth_heads(self.man["heads_npz"], C.SPEC["heads"]["scenario"],
                          C.SPEC["heads"]["last_month"])
        self.h, self.dates, self.grid = H["h"], H["dates"], H["grid"]
        self.mod = pd.DatetimeIndex(self.dates).to_period("M")
        self.zone = C.zone_map(self.grid, C.repo_path(C.SPEC["theta"]))
        self.truth = {}
        self.pairs = {}
        for t, v in self.man["truths"].items():
            tc = json.loads((synth_dir / f"truth_{t}" / "truth_column.json").read_text())
            self.truth[t] = tc["zonal"]
            panel = load_panel(v["data_dir"]).rename(columns={"x": "x_3826", "y": "y_3826"})
            for o in C.SPEC["origins"]:
                self.pairs[(t, o)] = self._pairs(panel, o)

    def month(self, t) -> int:
        return int(np.nonzero(self.mod == pd.Timestamp(t).to_period("M"))[0][0])

    def _pairs(self, panel: pd.DataFrame, origin: str) -> pd.DataFrame:
        pr = C.benchmark_pairs(panel, origin)
        pr["cell"] = [self.grid.cell_of(x, y) for x, y in zip(pr["x"], pr["y"], strict=True)]
        pr = pr[pr["cell"] >= 0].copy()
        pr["zone"] = self.zone[pr["cell"].to_numpy()]
        pr["ka"] = [self.month(t) for t in pr["ta"]]
        pr["kb"] = [self.month(t) for t in pr["tb"]]
        if (pr["ka"] <= self.month(pd.Period(origin, freq="M").start_time)).any():
            raise SystemExit("a scored survey precedes the origin")
        return pr.reset_index(drop=True)


def bound_flags(p: dict, form: str, frac: float) -> dict:
    """Distal slow parameters: value, recorded bounds (fit coordinates), at-bound flag."""
    s = p["zonal"][C.DISTAL]
    out = {}
    for k in SLOW_KEYS[form]:
        if k == "log_tau":
            floors = [float(p.get("tau_min_days") or 1.0)]
            floors += [float(v) for z, v in (p.get("tau_min_days_zone") or {}).items()
                       if z == "distal"]
            lo, hi = math.log(max(floors)), math.log(float(p["tau_ceiling_days"]))
        else:
            lo, hi = (float(v) for v in p["aquitard_bounds"][k])
        v = float(s[k])
        tol = frac * (hi - lo)
        out[k] = {"value": v, "bounds": [lo, hi], "tol": tol,
                  "at_bound": bool(v - lo <= tol or hi - v <= tol)}
    return out


def arm_metrics(w: World, p: dict, truth: str, form: str, origin: str) -> dict:
    fit = [dict(s) for s in p["zonal"]]
    tru = w.truth[truth]
    pr = w.pairs[(truth, origin)]
    ko = w.month(pd.Period(origin, freq="M").start_time)
    d = pr[pr["zone"] == C.DISTAL]
    cells, ka, kb = d["cell"].to_numpy(), d["ka"].to_numpy(), d["kb"].to_numpy()
    z = np.full(len(d), C.DISTAL)
    hd = w.h[cells]
    bg_fit = C.background_cm(fit, z, hd, ko, ka, kb, w.dates)
    bg_tru = C.background_cm(tru, z, hd, ko, ka, kb, w.dates)
    bg_fit1 = C.background_cm(fit, z, hd, ko, ka, kb, w.dates, offset_m=1.0)
    bg_tru1 = C.background_cm(tru, z, hd, ko, ka, kb, w.dates, offset_m=1.0)
    rows = np.arange(len(d))
    yrs = np.array([C.years_between(w.dates, a, b) for a, b in zip(ka, kb, strict=True)])
    slow_only = None
    if form in AQ_FORMS:
        aq = C.aquitard_series(fit[C.DISTAL], C.held(hd, ko))
        slow_only = float(((aq[rows, kb] - aq[rows, ka]) * 100.0 / yrs).mean())
    # next-year rates with the truth heads (reported): fit vs noise-free truth vs noisy pairs
    zall = pr["zone"].to_numpy()
    ha = w.h[pr["cell"].to_numpy()]
    cf = C.column_series(fit, zall, ha)
    ct = C.column_series(tru, zall, ha)
    r = np.arange(len(pr))
    ya = np.array([C.years_between(w.dates, a, b) for a, b in zip(pr["ka"], pr["kb"],
                                                                  strict=True)])
    mod = (cf[r, pr["kb"]] - cf[r, pr["ka"]]) * 100.0 / ya
    tr = (ct[r, pr["kb"]] - ct[r, pr["ka"]]) * 100.0 / ya
    obs = pr["obs"].to_numpy()
    dm = zall == C.DISTAL

    def rb(a, b, m):
        if not m.any():
            return {"n": 0}
        e = a[m] - b[m]
        return {"n": int(m.sum()), "rmse": float(np.sqrt((e ** 2).mean())),
                "bias": float(e.mean())}

    out = {
        "distal_pairs": int(len(d)),
        "background_fit_cm": float(bg_fit.mean()),
        "background_truth_cm": float(bg_tru.mean()),
        "background_ratio": (float(bg_fit.mean() / bg_tru.mean())
                             if abs(bg_tru.mean()) > 1e-9 else None),
        "empirical_sens_fit_cm_per_m": float((bg_fit - bg_fit1).mean()),
        "empirical_sens_truth_cm_per_m": float((bg_tru - bg_tru1).mean()),
        "slow_term_only_fit_cm": slow_only,
        "next_year_with_truth_heads": {
            "fit_vs_truth_distal": rb(mod, tr, dm), "fit_vs_truth_all": rb(mod, tr, dm | ~dm),
            "fit_vs_noisy_distal": rb(mod, obs, dm), "fit_vs_noisy_all": rb(mod, obs, dm | ~dm),
            "truth_vs_noisy_distal": rb(tr, obs, dm)},
        "bounds": bound_flags(p, form, G["at_bound_frac"]),
        "synthetic_leveling_r2": {"insample": p.get("r2_insample"),
                                  "outoffold": p.get("r2_outoffold")},
    }
    if form in AQ_FORMS:
        s = fit[C.DISTAL]
        sens = math.exp(float(s["log_ska"]) - float(s["log_tau_a"])) * 365.25
        out.update(sens_fit_per_yr=sens, sens_ratio=sens / C.SENS_TRUTH,
                   distal_aquitard={"ska": math.exp(float(s["log_ska"])),
                                    "tau_a_years": math.exp(float(s["log_tau_a"])) / 365.25,
                                    "h_a0_m": float(s["h_a0"])})
        m = fit[1]
        if "log_ska" in m:
            out["mid_aquitard"] = {"ska": math.exp(float(m["log_ska"])),
                                   "tau_a_years": math.exp(float(m["log_tau_a"])) / 365.25,
                                   "h_a0_m": float(m["h_a0"])}
    else:
        out["distal_vep"] = {"tau_years": math.exp(float(fit[C.DISTAL]["log_tau"])) / 365.25,
                             "h_pc0_m": float(fit[C.DISTAL]["h_pc0"])}
    return out


# ---------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------
def _all(vals):
    """True/False over both origins, None when any origin is not evaluable."""
    return None if any(v is None for v in vals) else bool(all(vals))


def gates(form: str, M: dict) -> dict:
    """``M[(truth, origin)]`` = metrics or None (not evaluable)."""
    lo, hi = G["rate_band"]
    slo, shi = G["sens_band"]
    origins = C.SPEC["origins"]

    def get(t, o):
        return M.get((t, o))

    a1 = _all([None if get("A", o) is None or get("A", o)["background_ratio"] is None
               else lo <= get("A", o)["background_ratio"] <= hi for o in origins])
    a3 = _all([None if get("A", o) is None
               else not any(b["at_bound"] for b in get("A", o)["bounds"].values())
               for o in origins])
    n = _all([None if get("N", o) is None
              else abs(get("N", o)["background_fit_cm"]) < G["n_abs_cm"] for o in origins])
    out = {"G0-A1": a1, "G0-A3": a3, "G0-N": n}
    if form in AQ_FORMS:
        out["G0-A2"] = _all([None if get("A", o) is None
                             else slo <= get("A", o)["sens_ratio"] <= shi for o in origins])
        out["G0-V"] = _all([None if get("V", o) is None
                            else get("V", o)["sens_ratio"] < G["v_spec_max_ratio"]
                            for o in origins])
    need = ["G0-A1", "G0-N", "G0-A3"] + (["G0-A2"] if form in AQ_FORMS else [])
    out["eligible_for_b"] = all(out[k] is True for k in need)
    out["not_evaluable"] = [k for k in need + (["G0-V"] if form in AQ_FORMS else [])
                            if out[k] is None]
    return out


def readiness(runs: Path, synth: Path, decl: dict) -> list[dict]:
    """Refuse to score unless the batch is finished and matches the declaration; returns the
    ledger rows."""
    fz = decl["frozen"]
    bad = []
    if runs.resolve() != C.repo_path(fz["runs_dir"]).resolve():
        bad.append(f"runs dir {runs} is not the declared {fz['runs_dir']}")
    if fz["spec_sha256"] != C.spec_sha256():
        bad.append("the spec differs from the declaration's")
    gm = synth / "generator_manifest.json"
    if not gm.exists() or C.file_sha256(gm) != fz["generator_manifest_sha256"]:
        bad.append("the generator manifest differs from the declaration's")
    tree = C.code_state()["tree_sha256"]
    if tree != fz["code"]["tree_sha256"]:
        bad.append(f"the evaluator runs on code tree {tree[:16]}, not the declared one "
                   "(run it through results/twin_runs/par32/py.sh)")
    if (runs / REPORT_NAME).exists():
        bad.append(f"{runs / REPORT_NAME} exists: gate 0 is scored once")
    rows = J.read_ledger(runs)
    if any(r["event"] == "evaluated" for r in rows):
        bad.append("the ledger already records an evaluation")
    lock = runs / LOCK_NAME
    if lock.exists():
        with open(lock, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
            except BlockingIOError:
                bad.append("run_e2.sh still holds its lock (batch running)")
    for a, *_ in C.all_arms():
        st = J.arm_status(rows, a)
        if st["attempts"] == 0:
            bad.append(f"{a}: never started")
        elif st["open"]:
            bad.append(f"{a}: attempt {st['open']} has not ended")
        elif len(st["completed"]) > 1:
            bad.append(f"{a}: {len(st['completed'])} completed attempts (scored once)")
    if bad:
        raise SystemExit("evaluation refused:\n  " + "\n  ".join(bad))
    return rows


def evaluate(runs: Path, synth: Path, decl: dict) -> dict:
    rows = readiness(runs, synth, decl)
    w = World(synth)
    report = {"spec_id": C.SPEC["id"], "spec_sha256": C.spec_sha256(),
              "synthetic_only": True, "declaration_created": decl["created"],
              "code_tree_sha256": decl["frozen"]["code"]["tree_sha256"],
              "generator": {k: w.man[k] for k in ("heads_sha256", "noise", "solved")},
              "arms": {}, "forms": {}}
    for form in C.SPEC["forms"]:
        M = {}
        for t in C.SPEC["primary_truths"]:
            for o in C.SPEC["origins"]:
                a = C.arm_name(t, form, o)
                d = runs / a
                st = J.arm_status(rows, a)
                rec = {"arm": a, "attempts": st["attempts"]}
                vj = d / J.COLUMN_JSON
                jj = d / "job.json"
                job = json.loads(jj.read_text()) if jj.exists() else None
                if not st["completed"] or not vj.exists():
                    rec["status"] = "failed"
                    M[(t, o)] = None
                else:
                    _, p = C.load_column_sets(vj)
                    bad = conformity(p, job, t, form, o, runs, w.man, decl, st)
                    if bad:
                        rec.update(status="non-conforming", reasons=bad)
                        M[(t, o)] = None
                    else:
                        rec.update(status="ok", metrics=arm_metrics(w, p, t, form, o))
                        M[(t, o)] = rec["metrics"]
                report["arms"][a] = rec
        report["forms"][form] = gates(form, M)
    elig = [f for f in C.SPEC["forms"] if report["forms"][f]["eligible_for_b"]]
    report["verdict"] = {
        "eligible_forms_for_I1.X5SC": elig,
        "b_runs": bool(elig),
        "b_primary": ("aq" if "aq" in elig else ("vt" if elig else None)),
        "level_dependent_response_statable": "aq" in elig,
        "aquitard_level_sensitivity_claimable": bool(
            report["forms"]["aq"]["eligible_for_b"] and report["forms"]["aq"]["G0-V"] is True),
        "rule": G["eligible_for_b"], "only_vt_passes": G["only_vt_passes"]}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(C.repo_path(C.RUNS_DIR)))
    ap.add_argument("--synth", default=str(C.repo_path(C.SYNTH_DIR)))
    ap.add_argument("--declaration", default=str(C.repo_path(C.DECL_PATH)))
    args = ap.parse_args(argv)
    runs, synth = Path(args.runs), Path(args.synth)
    decl = J.load_declaration(args.declaration)
    rep = evaluate(runs, synth, decl)
    out = runs / REPORT_NAME
    data = json.dumps(rep, indent=1, default=str)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(data)
    J.append(runs, {"event": "evaluated", "report": str(out),
                    "report_sha256": C.file_sha256(out)})
    for f, g in rep["forms"].items():
        print(f"{f:4s}: " + "  ".join(f"{k} {v}" for k, v in g.items()))
    print("verdict:", json.dumps(rep["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
