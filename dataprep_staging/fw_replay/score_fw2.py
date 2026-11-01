"""Score FW.weather_replay.v2: does a dry year expose the compaction column's permanent
(inelastic) channel? Model-only; 'energy+weather' mode.

v2 is FW.weather_replay.v1's recipe with the weather replayed (user ruling R7.fw_both,
rulings 2026-11-01f). v1 was registered pumping-only by the rain manifest's absolute-path
bug and runs as registered (R7.fw_v1_pumping, R7.fw_both); v1 is not withdrawn. The two are
separate, concurrent declarations.

Everything that scores is v1's scorer (score_fw.py), imported read-only and bound by sha256
in v2's frozen_code: arms, arm specs, file checks, gate check, reproduction, statistic,
thresholds and the wording function. v2 changes only:
  * the id, the scorer version and the frozen code (this file, run_fw2.sh, score_fw.py);
  * the replay mode, which must be 'energy+weather' (any other is refused);
  * its own files: results/twin_forward/fw_replay_v2/<arm>.{npz,fw.npz},
    results/twin_runs/fw_replay_v2/{attempts.jsonl,gate_forcing.json} and the report
    results/dataprep/reports/FW.weather_replay.v2/report.json; a path inside v1's
    directories is refused, so v2 never reads or writes v1's files;
  * the wordings: v1's texts with v2's id, and every text that cites a replayed-year result
    carries the zero-filled vendor-gauge caveat (GAUGE_CAVEAT: rf11 all of 2021-22, rf15 2015
    H2), since in this mode the replayed 2015, 2021 and 2022 rain carries that artefact.
The page run results/twin_forward/datum_gate_innov.npz is read for C's reproduction only.
No observation of any family is read. Order of work as v1 (score_fw.py docstring, steps 0-3);
``--repro`` runs steps 0-2 on C alone; ``--verify-registration`` step 0's registration only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import score_fw as S1  # noqa: E402  (v1's scorer, read-only; bound in FROZEN)

X = S1.X
V1_ID = S1.PREREG_ID
PREREG_ID = "FW.weather_replay.v2"
SCORER_VERSION = "2.0.0"
FIELDS = "energy+weather"                 # ruling R7.fw_both: v2 replays the weather
FROZEN = (f"{X}/score_fw2.py", f"{X}/run_fw2.sh", f"{X}/score_fw.py")
OUT_STEM = "results/twin_forward/fw_replay_v2/{arm}"
LOGDIR = "results/twin_runs/fw_replay_v2"
DEFAULT_ATTEMPTS = f"{LOGDIR}/attempts.jsonl"
GATE_JSON = f"{LOGDIR}/gate_forcing.json"
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"
TMUX_PREFIX = "fwv2_"                     # v1's runner uses fw1..fwN
V1_DIRS = (S1.OUT_STEM.format(arm="").rstrip("/"), S1.LOGDIR,
           f"results/dataprep/reports/{V1_ID}")
GAUGE_CAVEAT = ("Rain is the vendor cache the model was calibrated on, which zero-fills gauge "
                "rf11 for all of 2021-22 and rf15 in 2015 H2 (zero-filled, not dry), so the "
                "replayed 2015, 2021 and 2022 carry that artefact.")

# unchanged from v1 (one definition, imported)
ENGINE, ARMS, PRIMARY = S1.ENGINE, S1.ARMS, S1.PRIMARY
RED_DET, RED_RAW, RED_WEA, SEQ_STARTS = S1.RED_DET, S1.RED_RAW, S1.RED_WEA, S1.SEQ_STARTS
SCEN_NAMES, REF, REF_SHA256 = S1.SCEN_NAMES, S1.REF, S1.REF_SHA256
SHARE_THRESHOLD, MIN_AVOIDED_M, MIN_EVALUABLE = (S1.SHARE_THRESHOLD, S1.MIN_AVOIDED_M,
                                                 S1.MIN_EVALUABLE)
PLACEHOLDER_FILL = S1.PLACEHOLDER_FILL
EngineeringError = S1.EngineeringError
sha256, cyclic, years_text, arm_spec = S1.sha256, S1.cyclic, S1.years_text, S1.arm_spec
under_text = S1.under_text


def arms(fields: str = FIELDS) -> tuple[str, ...]:
    """Every arm v2 runs and scores (the weather-only RW arms included)."""
    if fields != FIELDS:
        raise ValueError(f"{PREREG_ID} runs in '{FIELDS}' mode only, not {fields!r}")
    return S1.arms(fields)


# ----------------------------------------------------------------------------- step 0
def load_registered(prereg_dir: Path | None = None, rows_dir: Path | None = None
                    ) -> tuple[dict, str]:
    sys.path.insert(0, str(REPO / "dataprep_staging"))
    from harness import prereg3

    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    try:
        return prereg3.load_prereg(PREREG_ID, **kw)
    except prereg3.PreregistrationError as err:
        raise EngineeringError(f"registration: {err}") from err


def check_frozen(doc: dict, repo: Path = REPO) -> dict:
    want = doc.get("frozen_code") or {}
    got = {p: sha256(repo / p) if (repo / p).exists() else "missing" for p in FROZEN}
    bad = {p: {"registered": want.get(p), "now": got[p]} for p in FROZEN
           if want.get(p) != got[p]}
    if bad:
        raise EngineeringError(f"frozen code differs from the registered declaration: {bad}")
    return got


def declared_fields(doc: dict) -> str:
    f = (doc.get("recipe") or {}).get("replay_fields")
    if f != FIELDS:
        raise EngineeringError(f"the declaration's recipe.replay_fields is {f!r}: {PREREG_ID} "
                               f"scores '{FIELDS}' only (ruling R7.fw_both)")
    return f


def check_attempts(attempts: Path, files: dict, gate: Path | None,
                   need_repro: bool = True) -> dict:
    """v1's check (score_fw.check_attempts) on v2's attempt log, with v2's refusal of a
    logged failed reproduction first."""
    failed = [r for r in S1._records(attempts) if r.get("event") == "repro"
              and int(r.get("rc", -1)) == 3]
    if failed:
        raise EngineeringError(f"a failed reproduction (exit 3) is logged ({len(failed)}x, "
                               f"first {failed[0].get('time')}): nothing more is scored under "
                               f"{PREREG_ID}")
    return S1.check_attempts(attempts, files, gate, need_repro)


def in_v1_dirs(p) -> bool:
    """True when ``p`` lies in one of v1's output, log or report directories."""
    q = rp(str(p)).resolve()
    for d in V1_DIRS:
        base = (REPO / d).resolve()
        if q == base or base in q.parents:
            return True
    return False


# ----------------------------------------------------------------------------- wording
def _v2(text: str, caveat: bool) -> str:
    t = text.replace(V1_ID, PREREG_ID)
    return f"{t} {GAUGE_CAVEAT}" if caveat else t


def wording(case: str, fill: dict, fields: str = FIELDS) -> dict:
    """v1's pre-declared wordings (score_fw.wording) in 'energy+weather' mode, with v2's id;
    every text cites a replayed-year result and so carries GAUGE_CAVEAT."""
    if fields != FIELDS:
        raise ValueError(f"{PREREG_ID} words '{FIELDS}' only, not {fields!r}")
    return {k: _v2(v, caveat=True) for k, v in S1.wording(case, fill, fields).items()}


def score(caps: dict, fields: str = FIELDS) -> dict:
    """v1's statistic, wording case and report-only tables (score_fw.score), unchanged;
    the wordings are v2's."""
    if fields != FIELDS:
        raise ValueError(f"{PREREG_ID} scores '{FIELDS}' only, not {fields!r}")
    out = S1.score(caps, fields)
    case = out["wording_case"]
    if case:
        out["wording"] = wording(case, S1.fill_of(out["summary"]), fields)
    else:                                    # not evaluable: no result is cited
        out["wording"] = {k: _v2(v, caveat=False) for k, v in out["wording"].items()}
    out["rain_caveat"] = GAUGE_CAVEAT
    return out


# ----------------------------------------------------------------------------- main
def rp(s: str) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def files_of(arm: str, root: str | None = None) -> tuple[Path, Path]:
    stem = OUT_STEM.format(arm=arm) if root is None else f"{root}/{arm}"
    return rp(stem + ".npz"), rp(stem + ".fw.npz")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", default=None, help="directory of the arms' files (default "
                    f"{OUT_STEM.format(arm='').rstrip('/')})")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--gate", default=GATE_JSON)
    ap.add_argument("--attempts", default=DEFAULT_ATTEMPTS)
    ap.add_argument("--out", default=None, help="report JSON (scoring mode; required there)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--repro", action="store_true", help="steps 0-2 on C alone")
    mode.add_argument("--verify-registration", action="store_true",
                      help="step 0's registration part only; reads no output")
    a = ap.parse_args(argv)
    files: dict = {}
    try:
        if not (a.verify_registration or a.repro) and not a.out:
            raise EngineeringError("scoring mode needs --out")
        v1 = [s for s in (a.outputs, a.gate, a.attempts, a.out) if s and in_v1_dirs(s)]
        if v1:
            raise EngineeringError(f"{v1} lie in {V1_ID}'s directories: {PREREG_ID} never "
                                   "reads or writes v1's files")
        if a.out and rp(a.out).exists():
            raise EngineeringError(f"{a.out} exists: the declaration is scored once")
        doc, decl_sha = load_registered()
        frozen = check_frozen(doc)
        fields = declared_fields(doc)
        engine = S1.declared_engine(doc)
        county = S1.declared_county(doc)
        digests = S1.check_input_digests(doc)
        files = {arm: files_of(arm, a.outputs) for arm in arms(fields)}
        reg = {"declaration_sha256": decl_sha, "frozen_code_sha256": frozen,
               "input_digests": digests, "replay_fields": fields, "engine_code": engine,
               "county_map": county, "arms": list(files)}
        print(f"step 0 ok: {PREREG_ID} registered (sha256 {decl_sha[:16]}), frozen code and "
              "declared inputs match")
        if a.verify_registration:
            return 0
        reg["gate"] = S1.check_gate_record(rp(a.gate), engine)
        if a.repro:
            reg["attempts"] = check_attempts(rp(a.attempts), {"C": files["C"]}, rp(a.gate),
                                             need_repro=False)
        else:
            reg["attempts"] = check_attempts(rp(a.attempts), files, rp(a.gate))
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 0): {err}", file=sys.stderr)
        return 1
    try:
        ref = S1.load_forward(rp(a.ref), "page", fields, S1.REF_NAMES, S1.SCEN_DESC[:3],
                              page=True)
        if ref["sha256"] != S1.REF_SHA256:
            raise EngineeringError(f"{a.ref} is not the registered page run")
        c = S1.load_forward(files["C"][0], "C", fields)
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    rep = S1.reproduction(c, ref)
    print(f"step 1 ok (C, page run); step 2 reproduction {'ok' if rep['ok'] else 'FAIL'}")
    for s, r in rep["scenarios"].items():
        print(f"  {s}: max |C - page| heads {r['heads_max_abs_diff_m']:.2e} m, subsidence "
              f"{r['subs_max_abs_diff_m']:.2e} m")
    if not rep["ok"]:
        print("ENGINEERING FAIL (step 2): C does not reproduce the page run; no other arm "
              "read", file=sys.stderr)
        return 3
    if a.repro:
        print(f"repro: ok (C sha256 {c['sha256']})")
        return 0
    try:
        fw = {"C": c, **{arm: S1.load_forward(files[arm][0], arm, fields)
                         for arm in files if arm != "C"}}
        caps = {arm: S1.load_capture(files[arm][1], arm, fields, engine, fw[arm], county)
                for arm in files}
    except EngineeringError as err:
        print(f"ENGINEERING FAIL (step 1): {err}", file=sys.stderr)
        return 2
    res = score(caps, fields)
    report = {"prereg_id": PREREG_ID, "scorer_version": SCORER_VERSION, "registration": reg,
              "reproduction": rep,
              "files": {arm: {"npz": {"path": fw[arm]["path"], "sha256": fw[arm]["sha256"]},
                              "capture": {"path": caps[arm]["path"],
                                          "sha256": caps[arm]["sha256"],
                                          "vs_forward_max_abs_m":
                                              caps[arm]["capture_vs_forward_max_abs_m"]}}
                        for arm in files}, **res}
    s = res["summary"]["share"]
    print(f"VERDICT: {res['verdict']} (wording case {res['wording_case']}; median inelastic "
          f"share {s['median']}, {res['summary']['n_evaluable']} of 11 evaluable)")
    p = rp(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:                      # never overwrite a scored report
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
